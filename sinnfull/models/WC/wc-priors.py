# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# # WC priors

from __future__ import annotations

# + tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup('theano')

# + tags=["hide-input"]
import numpy as np
import pymc3 as pm
import theano_shim as shim
from sinnfull.models.base import tag, Prior, PriorFactory
if __name__ == "__main__":
    from IPython.display import display
    from sinnfull.models._utils import truncated_histogram, sample_prior


# -

# :::{tip}
# The `Constant` distribution, although provided by PyMC3, was at some point [deprecated](https://github.com/pymc-devs/pymc3/pull/2452); it's not clear from the docs if it still is, but in any case it only accepts [integer values](https://github.com/pymc-devs/pymc3/issues/2451).  
# The better alternative is thus to use `Deterministic`, which has the benefit of not showing up as a variable to optimize.  
# :::

# ## Rich prior
#
# Based on the parameters used in Rich et al. (Scientific Reports, 2019).

@tag.rich
@PriorFactory
def WC_RichPrior(M:int):
    with Prior() as prior:
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        α = pm.Lognormal('α', np.log([100, 200]), 3, shape=(M,))
        β = pm.Lognormal('β', np.log(300.), 2, shape=(M,))
        # Separate sign and magnitude information of w
        A = np.concatenate((np.ones(M//2, dtype='int16'),
                            -np.ones(M//2, dtype='int16')))
        _w_mag = pm.Lognormal('_w',
                              mu=np.log([[ 1.60, 4.70],
                                         [ 3.00, 0.13]]),
                              sigma=3,
                              shape=(M,M))
        w = pm.Deterministic('w', A*_w_mag)
        h = pm.Normal('h', 0., 2, shape=(M,))
    return prior


# + tags=["remove-input"]
if __name__ == "__main__":
    prior = WC_RichPrior(2)
    display(prior)
    
    display(sample_prior(prior).cols(3))


# -

# ## Default prior
#
# Prior with values that are all near unity. Values are otherwise arbitrary.

@tag.zero_mean
@PriorFactory
def WC_Default(M:int):
    with Prior() as prior:
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        α = pm.Lognormal('α', -2, 3, shape=(M,))
        β = pm.Lognormal('β', 1, 2, shape=(M,))
        # Separate sign and magnitude information of w
        A = np.concatenate((np.ones(M//2, dtype='int16'),
                            -np.ones(M//2, dtype='int16')))
        _w_mag = pm.Lognormal('_w', -0.5, 3, shape=(M,M))
        w = pm.Deterministic('w', A*_w_mag)
        h = pm.Normal('h', 0., 2, shape=(M,))
    return prior


# + tags=["remove-input"]
if __name__ == "__main__":
    prior = WC_Default(2)
    display(prior)
    
    display(sample_prior(prior).cols(3))
# -

# ### Test
#
# Verify that `prior.random()` produces output which is compatible with the model parameters.
# (In the code below, `(4,1)` is an [RNG key](/sinnfull/rng).)

# + tags=["hide-input"]
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
# -


