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
# # OU priors

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup('theano')

# %% tags=["hide-input"]
import numpy as np
import pymc3 as pm
from sinnfull.models.base import tag, Prior
if __name__ == "__main__":
    from sinnfull.models._utils import truncated_histogram, sample_prior


# %% [markdown]
# ## Dale's law prior
# This prior imposes a sign on the columns of $\tilde{W}$, such that each process $\tilde{I}$ is either purely excitatory or purely inhibitory.
#
# Fixing the sign this way has some neuroscience justification in the form of Dale's law.

# %% tags=["hide-input"]
@tag('OU_AR', 'OU_FiniteNoise')  # Both forms of tagging
@tag.default                     # are equivalent
@tag.dale
def OU_DalePrior(M: int, Mtilde):
    with Prior() as prior:
        assert M % 2 == 0                   # Having odd M would imply arbitrarily choosing
        pm.Constant('M', M, dtype='int16')  # to have one extra positive W
        pm.Constant('Mtilde', Mtilde, dtype='int16')
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
    return prior


# %% tags=["hide-input"]
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
    from sinnfull.models.OU.OU import OU_AR, TimeAxis, Regularizer
    prior = OUInput_logτ_unsignedW_prior(2, 2)
    prior.random((4,1))        # Smoke test: `random()` works
    Prior.json_encoder(prior)  # Smoke test: serialization of prior

    time = TimeAxis(min=0, max=1, step=2**-5)
    model = OU_AR(time=time,
                  params=OU_AR.Parameters(**prior.random((4,1))),
                  rng=np.random.RandomState())

# %%
