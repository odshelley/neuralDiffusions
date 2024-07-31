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

Training SDEs as GANs was introduced in "Neural SDEs as Infinite-Dimensional GANs".
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
import numpy as np


###################
# First some standard helper objects.
###################

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]
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


###################
# Now we define the SDEs.
#
# We begin by defining the generator SDE.
###################
class GeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers, jump_intensity=1, has_jumps=True):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size
        self._jump_intensity = jump_intensity  # Rate parameter for the Poisson process
        self.has_jumps=has_jumps
        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        # If you have problems with very high drift/diffusions then consider scaling these so that they squash to e.g.
        # [-3, 3] rather than [-1, 1].
        ###################
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)
        self._jump_magnitude = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)


    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)


    def jump_magnitude(self, t, x):
        t = t.expand(x.size(0), 1)
        tz = torch.cat([t, x], dim=1)
        return self._jump_magnitude(tz)
    
    def jump_gradient(self, t, x):
        t = t.expand(x.size(0), 1)
        tz = torch.cat([t, x], dim=1)
        with torch.enable_grad():
            tz = tz.detach().requires_grad_()
            jump_mag = self._jump_magnitude(tz)
            # Initialize the Jacobian list
            jacobian_wrt_input = []
            # Compute the Jacobian
            for i in range(jump_mag.size(1)):  # Iterate over each output element
                grad_output = torch.zeros_like(jump_mag)
                grad_output[:, i] = 1
                gradients = torch.autograd.grad(
                    outputs=jump_mag,
                    inputs=tz,
                    grad_outputs=grad_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                jacobian_wrt_input.append(gradients[:, 1:].flatten())  # Exclude the gradient with respect to t

        # Stack the Jacobian rows to form the full Jacobian matrix
        jacobian_wrt_input = torch.stack(jacobian_wrt_input, dim=0)
        # with torch.enable_grad():
        #     tz = tz.detach().requires_grad_()
        #     jump_mag = self._jump_magnitude(tz)
        #     grad_z = torch.autograd.grad(jump_mag, tz, torch.ones_like(jump_mag), retain_graph=True)
        # deb=True
        # return grad_z[0][:, 1:]

        return jacobian_wrt_input

    def jump_gradient_wrt_theta(self, t, x):
        t = t.expand(x.size(0), 1)
        tz = torch.cat([t, x], dim=1)
        with torch.enable_grad():
            tz = tz.detach().requires_grad_()
            jump_mag = self._jump_magnitude(tz)
            # Initialize the Jacobian list
            jacobian_wrt_weights = []
            # Compute the Jacobian
            for i in range(jump_mag.size(1)):  # Iterate over each output element
                grad_output = torch.zeros_like(jump_mag)
                grad_output[:, i] = 1
                gradients = torch.autograd.grad(
                    outputs=jump_mag,
                    inputs=self._jump_magnitude.parameters(),
                    grad_outputs=grad_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )
                flattened_grads = torch.cat([grad.flatten() for grad in gradients])
                jacobian_wrt_weights.append(flattened_grads)

        # Stack the Jacobian rows to form the full Jacobian matrix
        jacobian_wrt_weights = torch.stack(jacobian_wrt_weights, dim=0)
   
        return jacobian_wrt_weights

    def jump_gradient_wrt_time(self, t, x):
        t = t.expand(x.size(0), 1)
        tz = torch.cat([t, x], dim=1)
        with torch.enable_grad():
            tz = tz.detach().requires_grad_()
            jump_mag = self._jump_magnitude(tz)
            grad_t = torch.autograd.grad(jump_mag, tz, torch.ones_like(jump_mag), retain_graph=True)
        # check if this is correct
        return grad_t[0][:, 0]  # Return gradient w.r.t. time only
    
    def jump_occurred_batch(self, t0, t1, batch_size):
        expected_jumps = self._jump_intensity * (t1 - t0).item()
        num_jumps = np.random.poisson([expected_jumps] * batch_size).tolist()
        num_jumps = torch.tensor(num_jumps, device='mps')
        
        jump_times_list = []
        for b in range(batch_size):
            num_jumps_b = int(num_jumps[b].item())
            if num_jumps_b > 0:
                jump_times = torch.sort(t0 + (t1 - t0) * torch.rand((num_jumps_b,), device='mps')).values
                jump_times_list.append(jump_times)
            else:
                jump_times_list.append(torch.tensor([], device='mps'))
        
        return jump_times_list

        
class Generator(torch.nn.Module):
    def __init__(
        self,
        data_size,
        initial_noise_size,
        noise_size,
        hidden_size,
        mlp_size,
        num_layers,
    ):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size
        self.data_size = 1
        self._initial = MLP(
            initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False
        )
        self._linear = torch.nn.Linear(hidden_size, 1)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)
        x0 = self._linear(x0)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(
            self._func,
            x0,
            ts,
            method="reversible_heun",
            dt=1.0,
            adjoint_method="adjoint_reversible_heun",
        )
        ys = xs.transpose(0, 1)

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
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
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
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=1.0,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()


###################
# Generate some data. For this example we generate some synthetic data from a time-dependent Ornstein-Uhlenbeck SDE.
###################
def get_data(batch_size, device):
    
    dataset_size = 10
    t_size = 64

    class PoissonProcessSimulator:
        def __init__(self, jump_intensity):
            self._jump_intensity = jump_intensity

        def jump_occurred_batch(self, t0, t1, batch_size):
            expected_jumps = self._jump_intensity * (t1 - t0).item()
            num_jumps = np.random.poisson([expected_jumps] * batch_size).tolist()
            num_jumps = torch.tensor(num_jumps, device=t0.device)

            jump_times_list = []
            for b in range(batch_size):
                num_jumps_b = int(num_jumps[b].item())
                if num_jumps_b > 0:
                    jump_times = torch.sort(t0 + (t1 - t0) * torch.rand((num_jumps_b,), device=t0.device)).values
                    jump_times_list.append(jump_times)
                else:
                    jump_times_list.append(torch.tensor([], device=t0.device))

            return jump_times_list

    poisson_simulator = PoissonProcessSimulator(jump_intensity=1)
    # Define time range
    t0 = torch.tensor(0.0, device=device)
    t1 = torch.tensor(float(t_size), device=device)

    # Simulate jump times for the Poisson process
    jump_times_list = poisson_simulator.jump_occurred_batch(t0, t1, dataset_size)

    # Convert jump times to Poisson process paths
    poisson_paths = []
    for jump_times in jump_times_list:
        path = torch.zeros(t_size, device=device)
        for jump_time in jump_times:
            path[int(jump_time.item()):] += .9
        poisson_paths.append(path)

    poisson_paths = torch.stack(poisson_paths)
    ts = torch.linspace(0, t_size - 1, t_size, device=device)
    ys = poisson_paths.unsqueeze(-1)

    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1),
                    ys], dim=2)
    # shape (dataset_size=1000, t_size=100, 1 + data_size=3)

    ###################
    # Package up.
    ###################
    data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return ts, data_size, dataloader


###################
# We'll plot some results at the end.
###################
def plot(ts, generator, dataloader, num_plot_samples, plot_locs):
    # Get samples
    for _ in range(10):

        real_samples, = next(iter(dataloader))
        assert num_plot_samples <= real_samples.size(0)
        real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
        real_samples = real_samples[..., 1]

        with torch.no_grad():
            generated_samples = generator(ts, real_samples.size(0)).cpu()
        generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
        generated_samples = generated_samples[..., 1]

        # Plot histograms
        # for prop in plot_locs:
        #     time = int(prop * (real_samples.size(1) - 1))
        #     real_samples_time = real_samples[:, time]
        #     generated_samples_time = generated_samples[:, time]
        #     _, bins, _ = plt.hist(real_samples_time.cpu().numpy(), bins=32, alpha=0.7, label='Real', color='dodgerblue',
        #                           density=True)
        #     bin_width = bins[1] - bins[0]
        #     num_bins = int((generated_samples_time.max() - generated_samples_time.min()).item() // bin_width)
        #     plt.hist(generated_samples_time.cpu().numpy(), bins=num_bins, alpha=0.7, label='Generated', color='crimson',
        #              density=True)
        #     plt.legend()
        #     plt.xlabel('Value')
        #     plt.ylabel('Density')
        #     plt.title(f'Marginal distribution at time {time}.')
        #     plt.tight_layout()
        #     plt.show()

        real_samples = real_samples[:num_plot_samples]
        generated_samples = generated_samples[:num_plot_samples]

        # Plot samples
        real_first = True
        generated_first = True
        for real_sample_ in real_samples:
            kwargs = {'label': 'Real'} if real_first else {}
            plt.plot(ts.detach().cpu().numpy(), real_sample_.detach().cpu().numpy(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
            real_first = False
        for generated_sample_ in generated_samples:
            kwargs = {'label': 'Generated'} if generated_first else {}
            plt.plot(ts.detach().cpu().numpy(), generated_sample_.detach().cpu().numpy(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
            generated_first = False
        # plt.legend()
        plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
        # Add gridlines with finer control
        plt.grid(True)

        plt.tight_layout()
    plt.savefig("pic.pdf")


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
        for real_samples, in dataloader:
            generated_samples = generator(ts, batch_size)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples)
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples


def main(
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        initial_noise_size=5,  # How many noise dimensions to sample at the start of the SDE.
        noise_size=3,          # How many dimensions the Brownian motion has.
        hidden_size=1,        # How big the hidden size of the generator SDE and the discriminator CDE are.
        mlp_size=16,           # How big the layers in the various MLPs are.
        num_layers=1,          # How many hidden layers to have in the various MLPs.

        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr=2e-4,      # Learning rate often needs careful tuning to the problem.
        discriminator_lr=1e-3,  # Learning rate often needs careful tuning to the problem.
        batch_size=1,        # Batch size.
        steps=2,            # How many steps to train both generator and discriminator for.
        init_mult1=3,           # Changing the initial parameter size can help.
        init_mult2=0.5,         #
        weight_decay=0.01,      # Weight decay.
        swa_step_start=500,    # When to start using stochastic weight averaging.

        # Evaluation and plotting hyperparameters
        steps_per_print=2,                   # How often to print the loss.
        num_plot_samples=1,                  # How many samples to use on the plots at the end.
        plot_locs=(0.1, 0.3, 0.5, 0.7, 0.9),  # Plot some marginal distributions at this proportion of the way along.
):
    # is_cuda = torch.cuda.is_available()
    # device = 'cuda' if is_cuda else 'cpu'
    # if not is_cuda:
    #     print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")
    device='mps'

    # Data
    ts, data_size, train_dataloader = get_data(batch_size=batch_size, device=device)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    # Models
    generator = Generator(data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers).to(device)
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
        for param in generator._initial.parameters():
            param *= init_mult1
        for param in generator._func.parameters():
            param *= init_mult2

    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)
    discriminator_optimiser = torch.optim.Adadelta(discriminator.parameters(), lr=discriminator_lr,
                                                   weight_decay=weight_decay)

    # Train both generator and discriminator.
    trange = tqdm.tqdm(range(steps))
    for step in trange:
        real_samples, = next(infinite_train_dataloader)

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
            total_unaveraged_loss = evaluate_loss(ts, batch_size, train_dataloader, generator, discriminator)
            if step > swa_step_start:
                total_averaged_loss = evaluate_loss(ts, batch_size, train_dataloader, averaged_generator.module,
                                                    averaged_discriminator.module)
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                             f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())

    _, _, test_dataloader = get_data(batch_size=batch_size, device=device)

    plot(ts, generator, test_dataloader, num_plot_samples, plot_locs)


if __name__ == '__main__':
    fire.Fire(main)