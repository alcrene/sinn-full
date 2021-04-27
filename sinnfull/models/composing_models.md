# Composing models

[These are notes to be used once we actually have a working composed model.]

We assume our models can be decomposed into *input*, *dynamics* and *observation* models:
\begin{align}
d\tilde{I}/dt &= F_{\text{in}}(t, \tilde{I}, \tilde{ξ}) \,, \\
du/dt &= F_{\text{dyn}}(t, \tilde{I}, u) \,,\\
d\bar{u}/dt &= F_{\text{obs}}(t, u, \bar{u}, \bar{ξ}) \,,
\end{align}
and that the $\bar{u}_t$ are observed. The important features here are that

1. The dynamics of $\tilde{I}$ are autonomous and generally stochastic.
2. The dynamics of $u$, once conditioned on $\tilde{I}$, are deterministic.
3. The dynamics of $\bar{u}$ may also be stochastic, but may only depend on $\tilde{I}$ through $u$.

Of these conditions, (1) and (3) make the problem harder because they force us to deal with latent variables, but represent what one typically faces with experimental data; (2) avoids having two latent noise sources which can compensate each other – it may not be strictly required.

The stochasticity on $\bar{u}$ is treated as observation noise, and leads to the usual likelihood terms for the model parameters.
The stochasticity on $\tilde{I}$ is treated as a latent variable, and we need to optimize each of its time points $\tilde{I}_t$. The initial conditions of $\tilde{I}$ and $u$ are treated as additional model parameters.

As long as the distributions induced by $F_{\text{in}}$, $F_{\text{dyn}}$ and $F_{\text{out}}$ are differentiable, so too is the log-likelihood, and we can attempt to optimize all parameters simultaneously. This *direct approach* is the motivation for the [*alternated SGD* algorithm](/sinnfull/optim/optimizers/alternated_sgd). Although simple, the need to alternate in practice between parameter and latent updates means that this algorithm has no convergence guarantees.

_Expectation-maximization_ (EM) is a usual approach to fitting models with latent variables that _does_ provide convergence guarantees. Such an algorithm requires a bit more infrastructure than the alternated SGD (specifically the possible to integrate a model backwards) and, while on our roadmap, is not yet part of this framework.
