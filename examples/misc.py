class OUGeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers, A_init=1.0, B_init=0.0):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        # Parameters for the mean-reverting drift
        self.A = torch.nn.Parameter(torch.tensor(A_init))
        self.B = torch.nn.Parameter(torch.tensor(B_init))

        # Diffusion remains the same MLP
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)

        # Enforce drift of the form A(B - X)
        drift = self.A * (self.B - x)

        # Compute diffusion
        diffusion = self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)

        return drift, diffusion
    
###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,
                                     adjoint_method='adjoint_reversible_heun',)
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


###################
# And that's (one way of doing) an SDE as a GAN. Have fun.
###################

###################
# Appendix: discriminators for a neural SDE
#
# This is a little long, but should all be quite straightforward. By the end of this you should have a comprehensive
# knowledge of how these things fit together.
#
# Let Y be the real/generated sample, and let H be the hidden state of the discriminator.
# For real data, then Y is some interpolation of an (irregular) time series. (As with neural CDEs, if you're familiar -
# for a nice exposition on this see https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py.)
# In the case of generated data, then Y is _either_ the continuous-time sample produced by sdeint, _or_ it is an
# interpolation (probably linear interpolation) of the generated sample between particular evaluation points, We'll
# refer to these as cases (*) and (**) respectively.
#
# In terms of the mathematics, our options for the discriminator are:
# (a1) Solve dH(t) = f(t, H(t)) dt + g(t, H(t)) dY(t),
# (a2) Solve dH(t) = (f, g)(t, H(t)) d(t, Y(t))
# (b) Solve dH(t) = f(t, H(t), Y(t)) dt.
# Option (a1) is what is stated in the paper "Neural SDE as Infinite-Dimensional GANs".
# Option (a2) is theoretically the same as (a1), but the drift and diffusion have been merged into a single function,
# and the sample Y has been augmented with time. This can sometimes be a more helpful way to think about things.
# Option (b) is a special case of the first two, by Appendix C of arXiv:2005.08926.
# [Note that just dH(t) = g(t, H(t)) dY(t) would _not_ be enough, by what's known as the tree-like equivalence property.
#  It's a bit technical, but the basic idea is that the discriminator wouldn't be able to tell how fast we traverse Y.
#  This is a really easy mistake to make; make sure you don't fall into it.]
#
# Whether we use (*) or (**), and (a1) or (a2) or (b), doesn't really affect the quality of the discriminator, as far as
# we know. However, these distinctions do affect how we solve them in terms of code. Depending on each combination, our
# options are to use a solver of the following types:
#
#      | (a1)   (a2)   (b)
# -----+----------------------
#  (*) | SDE           SDE
# (**) |        CDE    ODE
#
# So, (*) implies using an SDE solver: the continuous-time sample is only really available inside sdeint, so if we're
# going to use the continuous-time sample then we need to solve generator and discriminator together inside a single SDE
# solve. In this case, as our generator takes the form
# Y(t) = l(X(t)) with dX(t) = μ(t, X(t)) dt + σ(t, X(t)) dW(t),
# then
# dY(t) = l(X(t)) dX(t) = l(X(t))μ(t, X(t)) dt + l(X(t))σ(t, X(t)) dW(t).
# Then for (a1) we get
# dH(t) = ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) ) dt + g(t, H(t))l(X(t))σ(t, X(t)) dW(t),
# which we can now put together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t)                                )      ( σ(t, X(t))                  )
# d( Y(t) ) = ( l(X(t))μ(t, X(t)                         ) dt + ( l(X(t))σ(t, X(t))           ) dW(t)
#  ( H(t) )   ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) )      ( g(t, H(t))l(X(t))σ(t, X(t)) ),
# whilst for (b) we can put things together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t))       )      ( σ(t, X(t))        )
# d( Y(t) ) = ( l(X(t))μ(t, X(t) ) dt + ( l(X(t))σ(t, X(t)) ) dW(t)
#  ( H(t) )   ( f(t, H(t), Y(t)) )      ( 0                 )
#
# Phew, what a lot of stuff to write down. Don't be put off by this: there's no complicated algebra, it's literally just
# substituting one equation into another. Also, note that all of this is for the _generated_ data. If using real data,
# then Y(t) is as previously described always an interpolation of the data. If you're able to evaluate the derivative of
# the interpolation then you can then apply (a1) by rewriting it as dY(t) = (dY/dt)(t) dt and substituting in. If you're
# able to evaluate the interpolation itself then you can apply (b) directly.
#
# The benefit of using (*) is that everything can be done inside a single SDE solve, which is important if you're
# thinking about using adjoint methods and the like, for memory efficiency. The downside is that the code gets a bit
# more complicated: you need to be able to solve just the generator on its own (to produce samples at inference time),
# just the discriminator on its own (to evaluate the discriminator on the real data), and the combined
# generator-discriminator system (to evaluate the discriminator on the generated data).
#
# Right, let's move on to (**). In comparison, this is much simpler. We don't need to substitute in anything. We're just
# taking our generated data, sampling it at a bunch of points, and then doing some kind of interpolation (probably
# linear interpolation). Then we either solve (a2) directly with a CDE solver (regardless of whether we're using real or
# generated data), or solve (b) directly with an ODE solver (regardless of whether we're using real or generated data).
#
# The benefit of this is that it's much simpler to code: unlike (*) we can separate the generator and discriminator, and
# don't ever need to combine them. Also, real and generated data is treated the same in the discriminator. (Which is
# arguably a good thing anyway.) The downside is that we can't really take advantage of things like adjoint methods to
# backpropagate efficiently through the generator, because we need to produce (and thus store) our generated sample at
# lots of time points, which reduces the memory efficiency.
#
# Note that the use of ODE solvers for (**) is only valid because we're using _interpolated_ real or generated data,
# and we're assuming that we're using some kind of interpolation that is at least piecewise smooth. (For example, linear
# interpolation is piecewise smooth.) It wouldn't make sense to apply ODE solvers to some rough signal like Brownian
# motion - that's what case (*) and SDE solvers are about.
#
# Right, let's wrap up this wall of text. Here, we use option (**), (a2). This is arguably the simplest option, and
# is chosen as we'd like to keep the code readable in this example. To solve the CDEs we use the CDE solvers available
# through torchcde: https://github.com/patrick-kidger/torchcde.
###################
