# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # GWN objective functions

# %% tags=["remove-cell"]
from sinnfull.models.base import ObjectiveFunction, tag
from sinnfull.typing_ import IndexableNamespace


# %% [markdown]
# Objective functions must follow the usual limitations for being serializable: only use modules defined in `mackelab_toolbox.serialize.config.default_namespace` (these are listed in the cell below), or import additional modules within the function.
#

# %%
# Modules available in global scope when deserializing
import numpy as np
import math
import theano_shim as shim


# %% [markdown]
# ## Public interface
# All functions defined in this module are accessible from the collection `sinnfull.models.objectives.WC`.

# %% [markdown]
# ## Log-likelihood
#
# $$\begin{aligned}
# p(ξ_k | μ, σ) &= \log \left[\frac{1}{σ} \sqrt{\frac{Δt}{2 π}} \exp\left(-\frac{(ξ_k - μ)^2}{2σ\left/\sqrt{Δt}\right.} \right)\right] \\
# &= -\log σ + \tfrac{1}{2} \log Δt -\frac{(ξ_k - μ)^2}{2σ\left/\sqrt{Δt}\right.}
# \end{aligned}$$

# %%
@tag.GaussianWhiteNoise
@ObjectiveFunction(tags={'log L'})
def GWN_logp(model, k):
    "Log probability of the GaussianWhiteNoise model."
    μ=model.μ; logσ=model.logσ; σ=shim.exp(logσ);
    Δt=model.dt; ξ=model.ξ
    Δt = getattr(Δt, 'magnitude', Δt)  # In case 'Δt' is a Pint or Quantities
    norm = -logσ + 0.5*shim.log(Δt)
    gauss = - (ξ(k)-μ)**2 / (2*σ**2) * Δt
    return norm.sum() + gauss.sum()

# %% [markdown]
# > When the optimizer evaluates the objective, it is not known in general whether `ubar` or `u` are given or need to be computed. Hence we use round brackets instead of square brackets for indexing.

# %%
