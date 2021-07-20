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
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # WC priors

# %%
from __future__ import annotations

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup('theano')

# %% tags=["hide-input"]
import numpy as np
import pymc3 as pm
import theano_shim as shim
from sinnfull.models.base import tag, Prior
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
# ## Rich prior
#
# Based on the parameters used in Rich et al. (Scientific Reports, 2019).

# %%
@tag.rich
class WC_RichPrior(Prior):
    def __init__(self, M:int, scale=1., name="", model=None):
        """
        The `scale` argument should have a value > 0 and can be used to tighten
        or broaden the priors; it scales the standard deviation of each.
        """
        super().__init__(name=name, model=model)
        assert scale > 0, "`scale` argument must be greater than 0."
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        α = pm.Lognormal('α', np.log([100, 200]), 3*scale, shape=(M,))
        #β = pm.Lognormal('β', np.log(300.), 2*scale, shape=(M,))
        # Make β constant for fit stability (see [model notebook](./WC))
        # pm.Deterministic('β', shim.constant([300.]*M, dtype='float64'))
        pm.Deterministic('β', shim.constant([3.]*M, dtype='float64'))
        # Separate sign and magnitude information of w
        A = np.concatenate((np.ones(M//2, dtype='int16'),
                            -np.ones(M//2, dtype='int16')))
        _w_mag = pm.Lognormal('_w',
                              mu=np.log([[ 1.60, 4.70],
                                         [ 3.00, 0.13]]),
                              sigma=1.*scale,
                              shape=(M,M))
        w = pm.Deterministic('w', A*_w_mag)
        #h = pm.Normal('h', 0., 2*scale, shape=(M,))
        pm.Deterministic('h', shim.constant([0.]*M, dtype='float64'))


# %% tags=["remove-input"]
if __name__ == "__main__":
    prior = WC_RichPrior(2)
    display(prior)

    display(sample_prior(prior).cols(3))


# %% [markdown]
# ## Default prior
#
# Prior with values that are all near unity. Values are otherwise arbitrary.

# %%
@tag.default
class WC_Default(Prior):
    def __init__(self, M:int, name="", model=None):
        super().__init__(name=name, model=model)
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        α = pm.Lognormal('α', -2, 3, shape=(M,))
        β = pm.Lognormal('β', 1, 2, shape=(M,))
        # Separate sign and magnitude information of w
        A = np.concatenate((np.ones(M//2, dtype='int16'),
                            -np.ones(M//2, dtype='int16')))
        _w_mag = pm.Lognormal('_w', -0.5, 3, shape=(M,M))
        w = pm.Deterministic('w', A*_w_mag)
        h = pm.Normal('h', 0., 2, shape=(M,))


# %% tags=["remove-input"]
if __name__ == "__main__":
    prior = WC_Default(2)
    display(prior)

    display(sample_prior(prior).cols(3))

# %% [markdown]
# ### Test
#
# Verify that `prior.random()` produces output which is compatible with the model parameters.
# (In the code below, `(4,1)` is an [RNG key](/sinnfull/rng).)

# %% tags=["hide-input"]
if __name__ == "__main__":
    from sinnfull.models.WC.WC import WilsonCowan, TimeAxis
    prior = WC_RichPrior(2)
    prior.random((4,1))        # Smoke test: `random()` works
    Prior.json_encoder(prior)  # Smoke test: serialization of prior
    WilsonCowan.Parameters(**prior.random((4,1)))
        # Smoke test: generated prior is compatible with model

    prior = WC_Default(2)
    prior.random((4,1))        # Smoke test: `random()` works
    Prior.json_encoder(prior)  # Smoke test: serialization of prior
    WilsonCowan.Parameters(**prior.random((4,1)))
        # Smoke test: generated prior is compatible with model
