# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # GWN priors

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
# :::{admonition} Definining priors
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
# :::

# %% [markdown]
# :::{tip}
# The `Constant` distribution, although provided by PyMC3, was at some point [deprecated](https://github.com/pymc-devs/pymc3/pull/2452); it's not clear from the docs if it still is, but in any case it only accepts [integer values](https://github.com/pymc-devs/pymc3/issues/2451).  
# The better alternative is thus to use `Deterministic`, which has the benefit of not showing up as a variable to optimize.  
# :::

# %% [markdown]
# ## Default prior

# %%
@tag.default
class GWN_Prior(Prior):
    def __init__(self, M:int, mu_mean=0., mu_std=3., logsigma_mean=0., logsigma_std=1.5,
                 name="", model=None):
        super().__init__(name=name, model=model)
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        μ = pm.Normal('μ', mu_mean, mu_std, shape=(M,))
        logσ = pm.Normal('logσ', logsigma_mean, logsigma_std, shape=(M,))


# %% tags=["remove-input"]
if __name__ == "__main__":
    prior = GWN_Prior(2)
    display(prior)
    
    display(sample_prior(prior).cols(3))


# %% [markdown]
# ## Zero-mean prior
#
# This prior fixes the mean to 0.

# %%
@tag.zero_mean
class GWN_ZeroMeanPrior(Prior):
    def __init__(self, M:int, logsigma_mean=0., logsigma_std=1.,
                        name="", model=None):
        super().__init__(name=name, model=model)
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        pm.Deterministic('μ', shim.broadcast_to(
            shim.constant(0, dtype=shim.config.floatX), (M,)))
        logσ = pm.Normal('logσ', logsigma_mean, logsigma_std, shape=(M,))


# %% tags=["remove-input"]
if __name__ == "__main__":
    prior = GWN_ZeroMeanPrior(2)
    display(prior)
    
    display(sample_prior(prior).cols(3))

# %% [markdown]
# ### Test
#
# Verify that `prior.random()` produces output which is compatible with the model parameters.
# (In the code below, `(4,1)` is an [RNG key](/sinnfull/rng).)

# %% tags=["hide-input"]
if __name__ == "__main__":
    from sinnfull.models.GWN.GWN import GaussianWhiteNoise, TimeAxis
    prior = GWN_Prior(2)
    prior.random((4,1))        # Smoke test: `random()` works
    Prior.json_encoder(prior)  # Smoke test: serialization of prior
    GaussianWhiteNoise.Parameters(**prior.random((4,1)))
        # Smoke test: generated prior is compatible with model
    
    prior = GWN_ZeroMeanPrior(2)
    prior.random((4,1))
    Prior.json_encoder(prior)
    GaussianWhiteNoise.Parameters(**prior.random((4,1)))

# %%
