# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train an SDE as a GAN, on data from a time-dependent Ornstein--Uhlenbeck process.

Training SDEs as GANs was introduced in "Neural SDEs as Infinite-Dimensional GANs".d
https://arxiv.org/abs/2102.03657

This reproduces the toy example in Section 4.1 of that paper.

This additionally uses the improvements introduced in "Efficient and Accurate Gradients for Neural SDEs".
https://arxiv.org/abs/2105.13493

To run this file, first run the following to install extra requirements:
pip install fire
pip install git+https://github.com/patrick-kidger/torchcde.git

To run, execute:
python -m examples.sde_gan
"""
import fire
import matplotlib.pyplot as plt
import torch
import torch.optim.swa_utils as swa_utils
import torchcde
import torchsde
import tqdm

MU = 0.5
KAPPA = 2.0
THETA = 0.1
SIGMA = 0.3
RHO = 0.5
INITIAL_PRICE = 100
INITIAL_VOL = 0.1


###################
# First some standard helper objects.
###################


class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size), LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


class HestonGeneratorFunc(torch.nn.Module):
    sde_type = "stratonovich"
    noise_type = "general"

    def __init__(
        self,
        noise_size,
        hidden_size,
        mlp_size,
        num_layers,
        kappa=KAPPA,
        theta=THETA,
        rho=RHO,
        min_volatility=0.0001,
    ):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size
        self.register_buffer("kappa", torch.as_tensor(kappa))
        self.register_buffer("theta", torch.as_tensor(theta))
        self.register_buffer("rho", torch.as_tensor(rho))
        self._linear = torch.nn.Linear(hidden_size, 1)  # Maps MLP to be 1 dimensional
        self.min_volatility = min_volatility
        # Construct the Cholesky decomposition of the correlation matrix

        self._drift_mlp = MLP(
            1 + 1, hidden_size, mlp_size, num_layers, tanh=True
        )  # Drift MLP for price
        self._vol_of_vol_mlp = MLP(
            1 + 1, hidden_size, mlp_size, num_layers, tanh=True
        )  # MLP for volatility of volatility

    def f_and_g(self, t, x):
        # dS=self.mu * S
        # dV=self.kappa * (self.theta - V)
        # return torch.stack([dS, dV], dim=-1)
        # x consists of [S_t, v_t]
        S, V = x[:, 0:1], x[:, 1:2]
        t_expanded = t.expand(x.size(0), 1)
        tx_s = torch.cat([t_expanded, S], dim=1)  # (t, S_t)
        tx_v = torch.cat([t_expanded, V], dim=1)

        # Drift components
        drift_S = self._drift_mlp(tx_s)  # MLP output scaled by S
        drift_S = self._linear(drift_S) * S
        drift_v = self.kappa * (self.theta - V)

        # Ensure V is not below the minimum volatility threshold
        V = torch.clamp(V, min=self.min_volatility)

        # Diffusion components
        diffusion_S = torch.sqrt(V) * S
        diffusion_v = self._vol_of_vol_mlp(tx_v)
        diffusion_v = self._linear(diffusion_v) * torch.sqrt(V)

        # S, V=y[:, 0], y[:, 1]
        # sqrt_V=torch.sqrt(torch.clamp(V, min=0))

        # g1=torch.stack([S * sqrt_V, torch.zeros_like(S)], dim=-1)
        # g2=torch.stack([torch.zeros_like(S), self.sigma * sqrt_V], dim=-1)

        # g=torch.stack([g1, g2], dim=-1)

        # return g @ self.chol_matrix.T  # Apply the Cholesky decomposition

        # Construct the diffusion matrix G
        batch_size = S.size(0)
        zeros = torch.zeros(
            batch_size, 1, device=S.device
        )  # Changed shape to [batch_size, 1]
        sqrt_1_minus_rho2 = torch.sqrt(torch.tensor(1 - self.rho**2, device=S.device))
        G = torch.stack(
            [
                torch.cat([diffusion_S, self.rho * diffusion_v], dim=1),
                torch.cat([zeros, sqrt_1_minus_rho2 * diffusion_v], dim=1),
            ],
            dim=2,
        )

        drift = torch.cat([drift_S, drift_v], dim=1)
        return drift / 36, G / 6


class HestonGenerator(torch.nn.Module):
    def __init__(
        self,
        noise_size,
        data_size,  # needed?
        hidden_size,
        mlp_size,
        num_layers,
        initial_asset_price=INITIAL_PRICE,
        initial_variance=INITIAL_VOL,
        kappa=KAPPA,
        theta=THETA,
        rho=RHO,
    ):
        super().__init__()
        self.initial_asset_price = initial_asset_price
        self.initial_variance = initial_variance

        self._func = HestonGeneratorFunc(
            noise_size, hidden_size, mlp_size, num_layers, kappa, theta, rho
        )

    def forward(self, ts, batch_size):
        # Initial conditions
        S0 = torch.full((batch_size, 1), self.initial_asset_price, device=ts.device)
        v0 = torch.full((batch_size, 1), self.initial_variance, device=ts.device)
        x0 = torch.cat([S0, v0], dim=1)

        # Solve the SDE
        xs = torchsde.sdeint_adjoint(
            self._func,
            x0,
            ts,
            method="reversible_heun",
            dt=1.0,
            adjoint_method="adjoint_reversible_heun",
        )
        ys = xs.transpose(0, 1)
        # ys=self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
#
# There's actually a few different (roughly equivalent) ways of making the discriminator work. The curious reader is
# encouraged to have a read of the comment at the bottom of this file for an in-depth explanation.
###################
class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(
            1 + hidden_size,
            hidden_size * (1 + data_size),
            mlp_size,
            num_layers,
            tanh=True,
        )

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = MLP(
            1 + data_size, hidden_size, mlp_size, num_layers, tanh=False
        )
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(
            Y,
            self._func,
            h0,
            Y.interval,
            method="reversible_heun",
            backend="torchsde",
            dt=1.0,
            adjoint_method="adjoint_reversible_heun",
            adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()),
        )
        score = self._readout(hs[:, -1])
        return score.mean()


###################
# Generate some data. For this example we generate some synthetic data from a time-dependent Ornstein-Uhlenbeck SDE.
###################
def get_data(batch_size, device):
    dataset_size = 1000
    t_size = 36
    sqrt_t_size = 6

    class HestonSDE(torch.nn.Module):
        sde_type = "ito"
        noise_type = "general"

        def __init__(
            self,
            mu=MU,
            kappa=KAPPA,
            theta=THETA,
            sigma=SIGMA,
            rho=RHO,
            S0=INITIAL_PRICE,
            V0=INITIAL_VOL,
            device="mps",
        ):
            super().__init__()
            self.device = device
            self.register_buffer("mu", torch.as_tensor(mu))
            self.register_buffer("kappa", torch.as_tensor(kappa))
            self.register_buffer("theta", torch.as_tensor(theta))
            self.register_buffer("sigma", torch.as_tensor(sigma))
            self.register_buffer("rho", torch.as_tensor(rho))
            self.register_buffer("S0", torch.as_tensor(S0))
            self.register_buffer("V0", torch.as_tensor(V0))
            # Construct the Cholesky decomposition of the correlation matrix
            self.register_buffer(
                "chol_matrix",
                torch.linalg.cholesky(torch.tensor([[1.0, rho], [rho, 1.0]])),
            )

        def f(self, t, y):
            S, V = y[:, 0], y[:, 1]
            dS = self.mu * S
            dV = self.kappa * (self.theta - V)
            return torch.stack([dS, dV], dim=-1) / t_size

        def g(self, t, y):
            S, V = y[:, 0], y[:, 1]
            sqrt_V = torch.sqrt(torch.clamp(V, min=0))

            g1 = torch.stack([S * sqrt_V, torch.zeros_like(S)], dim=-1)
            g2 = torch.stack([torch.zeros_like(S), self.sigma * sqrt_V], dim=-1)

            g = torch.stack([g1, g2], dim=-1)

            return (
                g @ self.chol_matrix.T
            ) / sqrt_t_size  # Apply the Cholesky decomposition

    # Initial conditions
    y0 = torch.tensor([[INITIAL_PRICE, INITIAL_VOL]] * dataset_size, device=device)
    ts = torch.linspace(0, t_size - 1, t_size, device=device)

    # Create Heston SDE model
    heston_sde = HestonSDE().to(device)

    # Simulate paths
    ys = torchsde.sdeint(heston_sde, y0, ts, dt=1e-1)

    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    ys = torch.cat(
        [
            ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1),
            ys.transpose(0, 1),
        ],
        dim=2,
    )
    # shape (dataset_size=1000, t_size=100, 1 + data_size=3)

    ###################
    # Package up.
    ###################
    data_size = (
        ys.size(-1) - 1
    )  # How many channels the data has (not including time, hence the minus one).
    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    return ts, data_size, dataloader


###################
# We'll plot some results at the end.
###################
def plot_price(ts, generator, dataloader, num_plot_samples, plot_locs):
    # Get samples
    (real_samples,) = next(iter(dataloader))
    assert num_plot_samples <= real_samples.size(0)
    real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_price_samples = real_samples[..., 1]
    with torch.no_grad():
        generated_samples = generator(ts, real_samples.size(0)).cpu()
    generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
    generated_price_samples = generated_samples[..., 1]

    # Plot histograms
    for prop in plot_locs:
        time = int(prop * (real_price_samples.size(1) - 1))
        real_samples_time = real_price_samples[:, time]
        generated_samples_time = generated_price_samples[:, time]
        _, bins, _ = plt.hist(
            real_samples_time.cpu().numpy(),
            bins=32,
            alpha=0.7,
            label="Real",
            color="dodgerblue",
            density=True,
        )
        bin_width = bins[1] - bins[0]
        num_bins = int(
            (generated_samples_time.max() - generated_samples_time.min()).item()
            // bin_width
        )
        plt.hist(
            generated_samples_time.cpu().numpy(),
            bins=num_bins,
            alpha=0.7,
            label="Generated",
            color="crimson",
            density=True,
        )
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Marginal distribution at time {time}.")
        plt.tight_layout()
        plt.show()

    real_price_samples = real_price_samples[:num_plot_samples]
    generated_price_samples = generated_price_samples[:num_plot_samples]

    # Plot samples
    real_first = True
    generated_first = True
    for real_sample_ in real_price_samples:
        kwargs = {"label": "Real"} if real_first else {}
        plt.plot(
            ts.detach().cpu().numpy(),
            real_sample_.detach().cpu().numpy(),
            color="dodgerblue",
            linewidth=0.5,
            alpha=0.7,
            **kwargs,
        )
        real_first = False
    for generated_sample_ in generated_price_samples:
        kwargs = {"label": "Generated"} if generated_first else {}
        plt.plot(
            ts.detach().cpu().numpy(),
            generated_sample_.detach().cpu().numpy(),
            color="crimson",
            linewidth=0.5,
            alpha=0.7,
            **kwargs,
        )
        generated_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    plt.show()


def plot_vol(ts, generator, dataloader, num_plot_samples, plot_locs):
    # Get samples
    (real_samples,) = next(iter(dataloader))
    assert num_plot_samples <= real_samples.size(0)
    real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_price_samples = real_samples[..., 2]
    with torch.no_grad():
        generated_samples = generator(ts, real_samples.size(0)).cpu()
    generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
    generated_price_samples = generated_samples[..., 2]

    # Plot histograms
    for prop in plot_locs:
        time = int(prop * (real_price_samples.size(1) - 1))
        real_samples_time = real_price_samples[:, time]
        generated_samples_time = generated_price_samples[:, time]
        _, bins, _ = plt.hist(
            real_samples_time.cpu().numpy(),
            bins=32,
            alpha=0.7,
            label="Real",
            color="dodgerblue",
            density=True,
        )
        bin_width = bins[1] - bins[0]
        num_bins = int(
            (generated_samples_time.max() - generated_samples_time.min()).item()
            // bin_width
        )
        plt.hist(
            generated_samples_time.cpu().numpy(),
            bins=num_bins,
            alpha=0.7,
            label="Generated",
            color="crimson",
            density=True,
        )
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Marginal distribution at time {time}.")
        plt.tight_layout()
        plt.show()

    real_price_samples = real_price_samples[:num_plot_samples]
    generated_price_samples = generated_price_samples[:num_plot_samples]

    # Plot samples
    real_first = True
    generated_first = True
    for real_sample_ in real_price_samples:
        kwargs = {"label": "Real"} if real_first else {}
        plt.plot(
            ts.detach().cpu().numpy(),
            real_sample_.detach().cpu().numpy(),
            color="dodgerblue",
            linewidth=0.5,
            alpha=0.7,
            **kwargs,
        )
        real_first = False
    for generated_sample_ in generated_price_samples:
        kwargs = {"label": "Generated"} if generated_first else {}
        plt.plot(
            ts.detach().cpu().numpy(),
            generated_sample_.detach().cpu().numpy(),
            color="crimson",
            linewidth=0.5,
            alpha=0.7,
            **kwargs,
        )
        generated_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    plt.show()


###################
# Now do normal GAN training, and plot the results.
#
# GANs are famously tricky and SDEs trained as GANs are no exception. Hopefully you can learn from our experience and
# get these working faster than we did -- we found that several tricks were often helpful to get this working in a
# reasonable fashion:
# - Stochastic weight averaging (average out the oscillations in GAN training).
# - Weight decay (reduce the oscillations in GAN training).
# - Final tanh nonlinearities in the architectures of the vector fields, as above. (To avoid the model blowing up.)
# - Adadelta (interestingly seems to be a lot better than either SGD or Adam).
# - Choosing a good learning rate (always important).
# - Scaling the weights at initialisation to be roughly the right size (chosen through empirical trial-and-error).
###################


def evaluate_loss(ts, batch_size, dataloader, generator, discriminator):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for (real_samples,) in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples


def load_weights(
    generator,
    discriminator,
    generator_path="best_generator_heston.pth",
    discriminator_path="best_discriminator_heston.pth",
):
    generator.load_state_dict(torch.load(generator_path, weights_only=True))
    discriminator.load_state_dict(torch.load(discriminator_path, weights_only=True))


def main(
    # Architectural hyperparameters. These are quite small for illustrative purposes.
    initial_noise_size=5,  # How many noise dimensions to sample at the start of the SDE.
    noise_size=2,  # How many dimensions the Brownian motion has.
    hidden_size=16,  # How big the hidden size of the generator SDE and the discriminator CDE are.
    mlp_size=16,  # How big the layers in the various MLPs are.
    num_layers=1,  # How many hidden layers to have in the various MLPs.
    # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
    generator_lr=2e-3,  # Learning rate often needs careful tuning to the problem.
    discriminator_lr=1e-2,  # Learning rate often needs careful tuning to the problem.
    batch_size=1024,  # Batch size.
    steps=50,  # How many steps to train both generator and discriminator for.
    init_mult1=2,  # Changing the initial parameter size can help.
    init_mult2=1.0,  #
    weight_decay=0.01,  # Weight decay.
    swa_step_start=5000,  # When to start using stochastic weight averaging.
    # Evaluation and plotting hyperparameters
    steps_per_print=2,  # How often to print the loss.
    num_plot_samples=50,  # How many samples to use on the plots at the end.
    plot_locs=(
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    ),  # Plot some marginal distributions at this proportion of the way along.
    load_best_weights=True,  # Whether to load the best weights at the end of training
):
    # device="mps"
    # print('HELLO?')
    # TODO: change
    device='mps'
    # is_cuda = torch.cuda.is_available()
    # device = "cuda" if is_cuda else "cpu"
    # if not is_cuda:
    #     print(
    #         "Warning: CUDA not available; falling back to CPU but this is likely to be very slow."
    #     )

    # Data
    ts, data_size, train_dataloader = get_data(batch_size=batch_size, device=device)
    infinite_train_dataloader = (
        elem for it in iter(lambda: train_dataloader, None) for elem in it
    )

    # Models
    generator = HestonGenerator(
        noise_size, data_size, hidden_size, mlp_size, num_layers
    ).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers).to(
        device
    )
    # Weight averaging really helps with GAN training.
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    # Picking a good initialisation is important!
    # In this case these were picked by making the parameters for the t=0 part of the generator be roughly the right
    # size that the untrained t=0 distribution has a similar variance to the t=0 data distribution.
    # Then the func parameters were adjusted so that the t>0 distribution looked like it had about the right variance.
    # What we're doing here is very crude -- one can definitely imagine smarter ways of doing things.
    # (e.g. pretraining the t=0 distribution)
    with torch.no_grad():
        # for param in generator._initial.parameters():
        #     param *=init_mult1
        for param in generator._func.parameters():
            param *= init_mult2

    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimiser = torch.optim.Adadelta(
        generator.parameters(), lr=generator_lr, weight_decay=weight_decay
    )
    discriminator_optimiser = torch.optim.Adadelta(
        discriminator.parameters(), lr=discriminator_lr, weight_decay=weight_decay
    )

    # Track the lowest loss and corresponding steps
    lowest_loss = float("inf")
    best_loss_steps = []

    # Train both generator and discriminator.
    trange = tqdm.tqdm(range(steps))
    for step in trange:
        (real_samples,) = next(infinite_train_dataloader)

        generated_samples = generator(ts, batch_size)
        generated_score = discriminator(generated_samples)
        real_score = discriminator(real_samples)
        loss = generated_score - real_score
        loss.backward()

        for param in generator.parameters():
            param.grad *= -1
        generator_optimiser.step()
        discriminator_optimiser.step()
        generator_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

        ###################
        # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
        # LipSwish activation functions).
        ###################
        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

        # Stochastic weight averaging typically improves performance.
        if step > swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_unaveraged_loss = evaluate_loss(
                ts, batch_size, train_dataloader, generator, discriminator
            )
            if step > swa_step_start:
                total_averaged_loss = evaluate_loss(
                    ts,
                    batch_size,
                    train_dataloader,
                    averaged_generator.module,
                    averaged_discriminator.module,
                )
                trange.write(
                    f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                    f"Loss (averaged): {total_averaged_loss:.4f}"
                )

                # Save the model weights if the averaged loss is the lowest
                if total_averaged_loss < lowest_loss:
                    lowest_loss = total_averaged_loss
                    torch.save(
                        averaged_generator.module.state_dict(),
                        "best_generator_heston.pth",
                    )
                    torch.save(
                        averaged_discriminator.module.state_dict(),
                        "best_discriminator_heston.pth",
                    )
                    trange.write(
                        f"New best model found at step {step}, saving model weights."
                    )
                    # best_loss_steps.append(step)
            else:
                trange.write(
                    f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}"
                )

                # Save the model weights if the unaveraged loss is the lowest
                if total_unaveraged_loss < lowest_loss:
                    lowest_loss = total_unaveraged_loss
                    torch.save(generator.state_dict(), "best_generator_heston.pth")
                    torch.save(
                        discriminator.state_dict(), "best_discriminator_heston.pth"
                    )
                    trange.write(
                        f"New best model found at step {step}, saving model weights."
                    )
                    # best_loss_steps.append(step)

    # Load the best weights if the flag is set
    if load_best_weights:
        load_weights(generator, discriminator)

    # Set plot locations to be at the best loss locations
    # if best_loss_steps:
    #     best_loss_steps=sorted(best_loss_steps)
    #     plot_locs=[step / steps for step in best_loss_steps]

    time, dimension_of_SDE, test_dataloader = get_data(
        batch_size=batch_size, device=device
    )

    plot_price(ts, generator, test_dataloader, num_plot_samples, plot_locs)
    plot_vol(ts, generator, test_dataloader, num_plot_samples, plot_locs)


if __name__ == "__main__":
    fire.Fire(main)
