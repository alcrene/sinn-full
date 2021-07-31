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
# # OU objective functions

# %% tags=["remove-input"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup()

# %% tags=["hide-input"]
import numpy as np
import theano_shim as shim

from sinnfull.models.base import ObjectiveFunction, tag
from sinnfull.typing_ import IndexableNamespace
# from sinnfull.tags import TagDecorator

# %% [markdown]
# Objective functions must follow the usual limitations for being serializable: only use modules defined in `mackelab_toolbox.serialize.config.default_namespace` (these are listed in the cell below), or import additional modules within the function.

# %%
# Modules available in global scope when deserializing
import numpy as np
import math
import theano_shim as shim

# %% [markdown]
# ## Public interface
#
# All functions defined in this module are accessible from the collection `sinnfull.models.objectives.OU`.

# %% [markdown]
# ## OU – AR(1) model

# %% [markdown]
# ### Ad-hoc observation loss

# %%
@tag.OU_AR
@tag.OU_FiniteNoise
@ObjectiveFunction(tags={'nodyn', 'se'})
def OU_obs_se(model, k):
    "Squared error loss"
    Itilde=model.Itilde; I=model.I
    Wtilde=model.Wtilde; A=model.A
    sqrerr = ((shim.dot(A*Wtilde, Itilde(k)) - I(k))**2).sum()
    return -sqrerr

@tag.OU_AR
@tag.OU_FiniteNoise
@ObjectiveFunction(tags={'nodyn', 'l1'})
def OU_obs_l1(model, k):
    "L1 (absolute difference) loss"
    Itilde=model.Itilde; I=model.I
    Wtilde=model.Wtilde; A=model.A
    l1err = abs(shim.dot(A*Wtilde, Itilde(k)) - I(k)).sum()
    return -l1err


# %% [markdown]
# ### Forward log-likelihood

# %%
@tag.OU_AR
@ObjectiveFunction(tags={'forward'})
def OUAR_logp_forward(model, k):
    Δt = model.dt
    Δt = getattr(Δt, 'magnitude', Δt)
    μtilde=model.μtilde; σtilde=model.σtilde; τtilde=model.τtilde
    Itilde=model.Itilde; I=model.I
    norm_Ik = shim.log(σtilde*shim.sqrt(2*Δt)).sum()
    gauss_Ik = ((Itilde(k) - Itilde(k-1) + (Itilde(k-1)-μtilde)*Δt/τtilde)**2 / (4*σtilde**2*Δt)).sum()
    return - norm_Ik - gauss_Ik


# %% [markdown]
# ### Backward log-likelihood
#
# Dynamics are time-reversal invariant, so we can substitute $t \to -t$ into the expression for $d\tilde{I}^m$ (see [OU models](./OU)) to obtain an alternative expression which depends on future values of $\tilde{I}$.
#
# :::{note}
# To use this objective, the optimizer needs to maintain a flipped copy of the model, which is effectively integrated backwards.
#
# The included [AlternatedOptimizer](/sinnfull/optim/optimizers/alternated) does not support this.
# :::

# %%
@tag.OU_AR
@ObjectiveFunction(tags={'backward'})
def OUAR_logp_backward(model, k):
    Δt = model.dt
    Δt = getattr(Δt, 'magnitude', Δt)
    μtilde=model.μtilde; σtilde=model.σtilde; τtilde=model.τtilde
    Itilde=model.Itilde; I=model.I
    norm_Ik = shim.log(σtilde*shim.sqrt(2*Δt)).sum()
    gauss_Ik = ((Itilde(k) - Itilde(k+1) - (Itilde(k+1)-μtilde)*Δt/τtilde)**2 / (4*σtilde**2*Δt)).sum()
    return - norm_Ik - gauss_Ik


# %% [markdown]
# ### Smoothing log-likelihood
#
# A smoother is constructed by adding forward and backward filters, producing a time-symmetric objective. The difference is the same as that between a Kalman filter and a Kalman smoother.

# %%
OUAR_logp_smoother = (OUAR_logp_forward + OUAR_logp_backward) / 2


# %% [markdown]
# ## OU Finite Noise model
#
# The difference with functions of [OU – AR](#ou-ar1) is that we scale $σ$ by $\sqrt{τ}$ in the SDE.

# %%
@tag.OU_FiniteNoise
@ObjectiveFunction(tags={'forward'})
def OUFN_logp_forward(model, k):
    Δt = model.dt
    Δt = getattr(Δt, 'magnitude', Δt)
    μtilde=model.μtilde; σtilde=model.σtilde; τtilde=model.τtilde
    Itilde=model.Itilde; I=model.I
    norm_Ik = shim.log(σtilde*shim.sqrt(2*Δt/τtilde)).sum()
    gauss_Ik = ((Itilde(k) - Itilde(k-1) + (Itilde(k-1)-μtilde)*Δt/τtilde)**2 / (4*σtilde**2*Δt)).sum()
    return - norm_Ik - gauss_Ik


# %% [markdown]
# ### Backward log-likelihood

# %%
@tag.OU_FiniteNoise
@ObjectiveFunction(tags={'backward', 'se'})
def OUFN_logp_backward(model, k):
    Δt = model.dt
    Δt = getattr(Δt, 'magnitude', Δt)
    μtilde=model.μtilde; σtilde=model.σtilde; τtilde=model.τtilde
    Itilde=model.Itilde; I=model.I
    norm_Ik = shim.log(σtilde*shim.sqrt(2*Δt/τtilde)).sum()
    gauss_Ik = ((Itilde(k) - Itilde(k+1) - (Itilde(k+1)-μtilde)*Δt/τtilde)**2 / (4*σtilde**2*Δt)).sum()
    return - norm_Ik - gauss_Ik


# %% [markdown]
# ### Smoothing log-likelihood

# %%
OUFN_logp_smoother = (OUFN_logp_forward + OUFN_logp_backward) / 2

# %% [markdown]
# > When the optimizer evaluates the objective, it is not known in general whether `ubar` or `u` are given or need to be computed. Hence we use round brackets instead of square brackets for indexing.
