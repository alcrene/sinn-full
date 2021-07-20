# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (sinnfull)
#     language: python
#     name: sinnfull
# ---

# %% [markdown]
# # GaussObs priors

# %%
from __future__ import annotations

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup('theano')

# %% tags=["hide-input"]
import numpy as np
import pymc3 as pm
from sinnfull.models.base import tag, Prior
import theano_shim as shim
if __name__ == "__main__":
    from IPython.display import display
    from sinnfull.models._utils import truncated_histogram, sample_prior


# %% [markdown]
# ::::{admonition} Definining priors  
# :class: dropdown
#
# Priors are defined by defining a [_custom PyMC3 model_](http://docs.pymc.io/api/model.html#pymc3.model.Model). Effectively this means that
#
# - Priors should inherit from `sinnfull.models.Prior` (which itself inherits from `pymc3.Model`).
# - Prior distributions must be defined within an `__init__` method.
# - The `__init__` method must accept `name` and `model` arguments, and pass them to `super().__init__`.
# - The call `super().__init__` should be at the top.
#
# This means that prior definitions should look like the following:
#
# ```python
# class CustomPrior(Prior):
#     def __init__(self, prior_args, ..., name="", model=None):
#         super().__init__(name=name, model=model)
#         # Define prior distributions as usual below; e.g.
#         a = pm.Normal('a')
#         ...
# ```
#
# :::{tip}
# The `Constant` distribution, although provided by PyMC3, was at some point [deprecated](https://github.com/pymc-devs/pymc3/pull/2452); it's not clear from the docs if it still is, but in any case it only accepts [integer values](https://github.com/pymc-devs/pymc3/issues/2451).  
# The better alternative is thus to use `Deterministic`, which has the benefit of not showing up as a variable to optimize.  
# :::
#
# ::::

# %% [markdown]
# The Gaussian observation has the form (see Eq. [./GaussObs.py](eq:gauss-obs))
#
# $$\begin{gathered}
# \bar{u}_k \sim \mathcal{N}\left(\bar{W} u_k, \bar{Σ}\right) \,; \\
# \begin{aligned}[t]
# \bar{W} &\in \mathbb{R}^{C\times N} \,, & \bar{Σ} &\in \mathbb{R}_+^C
# \end{aligned}
# \,.\end{gathered}$$
#
# |**Model parameter**| Identifier | Description |
# |--|--|--
# |$\bar{W}$| `Wbar` | linear projection dynamic to observed variables |
# |$\bar{Σ} \in \mathbb{R}_+^C$| `Σbar` | variance of the observation noise |
# |$M$ | `M` | no. of dynamic components |
# |$C$ | `C` | number of observed channels |
#
# Different priors impose different constraints on $\bar{W}$.

# %% [markdown]
# ## Independent observations
# $$\begin{aligned}
# C &= M \\
# \bar{W} &= \begin{pmatrix}
# \bar{W}_{11} & 0  &  \\
# 0 & \bar{W}_{22} &  & \\
#  &  & \ddots & \\
#  &  &  &  \bar{W}_{NN}
# \end{pmatrix} \\
# \bar{W}_{ii} &= 1 \\
# \log \bar{Σ} &\sim \mathcal{N}\bigl(μ_{\bar{Σ}}, Σ_\bar{Σ}\bigr)
# \end{aligned}$$
# As a special case, when $\texttt{logvar_std} = 0$, $\log \bar{Σ}$ is replaced by a deterministic variable equal to $e^{\texttt{logvar_mean}}$.

# %%
@tag.independent
class GaussObs_IndepPrior(Prior):
    def __init__(self, C:int, M:int, logvar_mean=1, logvar_std=1,
                 name="", model=None):
        super().__init__(name=name, model=model)
        assert C == M, "The 'independent observations' prior requires that the number of channels (C) equals the number of variables (N)"
        pm.Deterministic('C', shim.constant(C, dtype='int16'))
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        pm.Deterministic('Wbar', shim.constant(np.eye(C), dtype=shim.config.floatX))
            # Identity matrix has no rounding errors, so we use the dtype that
            # will trigger the fewest errors/warnings
        if logvar_std == 0:
            Σbar = np.ones(C)*np.exp(logvar_mean)
            assert Σbar.shape == (C,)
            pm.Deterministic('Σbar', shim.constant(Σbar, dtype=shim.config.floatX))
        else:
            pm.Lognormal('Σbar', mu=logvar_mean, sigma=logvar_std,
                         shape=(C,))


# %% tags=["remove-input"]
if __name__ == "__main__":
    prior = GaussObs_IndepPrior(2, 2)
    display(prior)
    
    display(sample_prior(prior).cols(3))

# %% tags=["remove-input"]
if __name__ == "__main__":
    prior = GaussObs_IndepPrior(2, 2, logvar_std=0)
    display(prior)
    
    display(sample_prior(prior).cols(3))

# %%
