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
# :::{tip}
# The `Constant` distribution, although provided by PyMC3, was at some point [deprecated](https://github.com/pymc-devs/pymc3/pull/2452); it's not clear from the docs if it still is, but in any case it only accepts [integer values](https://github.com/pymc-devs/pymc3/issues/2451).  
# The better alternative is thus to use `Deterministic`, which has the benefit of not showing up as a variable to optimize.  
# :::

# %% [markdown]
# ## Default prior

# %%
@tag.default
def GWN_Prior(M:int, mu_mean=0., mu_std=1., logsigma_mean=0., logsigma_std=1.):
    with Prior() as prior:
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        μ = pm.Normal('μ', mu_mean, mu_std, shape=(M,))
        logσ = pm.Normal('logσ', logsigma_mean, logsigma_std, shape=(M,))
    return prior


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
def GWN_ZeroMeanPrior(M:int, logsigma_mean=0., logsigma_std=1.):
    with Prior() as prior:
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        pm.Deterministic('μ', shim.broadcast_to(
            shim.constant(0, dtype=shim.config.floatX), (M,)))
        logσ = pm.Normal('logσ', logsigma_mean, logsigma_std, shape=(M,))
    return prior


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
