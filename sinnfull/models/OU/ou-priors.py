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
# # OU priors

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup('theano')

# %% tags=["remove-input"]
import numpy as np
import pymc3 as pm
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
# ## Dale's law prior
# This prior imposes a sign on the columns of $\tilde{W}$, such that each process $\tilde{I}$ is either purely excitatory or purely inhibitory.
#
# Fixing the sign this way has some neuroscience justification in the form of Dale's law.

# %%
@tag('OU_AR', 'OU_FiniteNoise')  # Both forms of tagging
@tag.default                     # are equivalent
@tag.dale
class OU_DalePrior(Prior):
    def __init__(self, M: int, Mtilde, name="", model=None):
        super().__init__(name=name, model=model)
        assert M % 2 == 0                   
            # Having odd M would imply having one extra positive W
        pm.Deterministic('M', shim.constant(M, dtype='int16'))
        pm.Deterministic('Mtilde', shim.constant(tilde, dtype='int16'))
        # Create sign array, with half +1, half -1
        # Each column of A*Wtilde has the same sign (column = input latent process)
        A = np.concatenate((np.ones(Mtilde//2, dtype='int16'),
                            -np.ones(Mtilde//2, dtype='int16')))
        A = pm.Constant('A', A, shape=(Mtilde,), dtype=A.dtype)
        μ = pm.Normal('μtilde', mu=0, sigma=2, shape=(Mtilde,))
        logτtilde = pm.Normal('logτtilde', mu=0, sigma=1, shape=(Mtilde,))
        _Wtilde_transp = pm.Dirichlet('_Wtilde_transp', a=np.ones((M,Mtilde)))
        Wtilde = pm.Deterministic('Wtilde', _Wtilde_transp.T)
        logσtilde = pm.Normal('logσtilde', mu=2, sigma=2, shape=(Mtilde,))


# %% tags=["remove-input"]
if __name__ == "__main__":
    prior = OU_DalePrior(2, 2)
    display(prior)

    # Verify that all columns sum to one
    assert np.isclose(prior.random((3,))['Wtilde'].sum(axis=0), 1).all()

    display(sample_prior(prior))

# %% [markdown]
# ### Test
#
# Verify that `prior.random()` produces output which is compatible with the model parameters.
# (In the code below, `(4,1)` is an [RNG key](/sinnfull/rng).)

# %% tags=["hide-input"]
if __name__ == "__main__":
    from sinnfull.models.OU.OU import OU_AR, TimeAxis
    prior = OU_DalePrior(2, 2)
    prior.random((4,1))        # Smoke test: `random()` works
    Prior.json_encoder(prior)  # Smoke test: serialization of prior

    OU_AR.Parameters(**prior.random((4,1)))
        # Smoke test: generated prior is compatible with model

# %%
