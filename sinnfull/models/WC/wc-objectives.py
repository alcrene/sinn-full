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
# # WC objective functions

# %%
import numpy as np
import theano_shim as shim

from sinnfull.models.base import ObjectiveFunction, tag
from sinnfull.typing_ import IndexableNamespace


# %% [markdown]
# Objective functions must follow the usual limitations for being serializable: only use modules defined in `mackelab_toolbox.serialize.config.default_namespace` (these are listed in the cell below), or import additional modules within the function.
#

# %% [markdown]
# :::{warning}  
# Because the Wilson-Cowan model is deterministic, it does not define a likelihood. The objectives defined here are therefore all *ad hoc*: they are convex and their maximum coincides with the true parameters, such that optimizing them allows to recover the model parameters. However:
#
# - *They cannot be combined with other objectives*. This is because, whereas two likelihoods for $θ$ (parameters) and $η$ (latents) can factorize as  
#   $\log p(Θ, η | \text{data}) = \log p(\text{data} | Θ, η) + \log p(η | Θ) + \log p(θ) + C\,,$  
#   the relative weighting of the objectives $\log p(\text{data} | Θ, η)$, $\log p(η | Θ)$, $\log p(θ)$ is essential for the MAP to remain correct. This relative weighting is not provided with an ad hoc objective.
# - Perhaps obviously, they cannot be interpreted as probabilities. With a likelihood, the difference between the likelihoods of two parameter sets tells us how much more likely one is compared to the other. With an ad hoc objective, we can only say that one is a better fit than the other. (Strictly speaking, it is even worse: since only the maximum is meaningful, even just comparing the value of the objective at other points is unsafe.)
#
# :::

# %%
# Modules available in global scope when deserializing
import numpy as np
import math
import theano_shim as shim


# %% [markdown]
# ## Public interface
# All objective functions defined in this module are accessible from the collection `sinnfull.models.objectives.WC`.

# %% [markdown]
# ## Ad-hoc squared-error objective
#
# A.k.a. a Gaussian objective, since it corresponds to the log of a Gaussian (plus constant terms).
#
# $$l(\hat{u}_k; s) = -\left(\frac{\hat{u}_k - u_k}{s}\right)^2$$

# %%
@tag.WilsonCowan
@ObjectiveFunction(tags={'se', 'gaussian', 'forward'})
def WC_se(model, k, s=1):
    "Squared error objective"
    u_predict = model.u_upd(model, k)
    return -(((u_predict - model.u[k])/s)**2).sum()


# %% [markdown]
# ## Ad-hoc Lorentz objective
#
# Corresponds to the log of a Lorentz (also known as Cauchy or Breit-Wigner) distribution. The motivation is that with its heavy tails, a Lorentz should not pull outliers towards its mean as strongly as a Gaussian would. As with the Gaussian, we neglect constant terms.
#
# $$l(\hat{u}_k; s) = -\log\left[1 + \left(\frac{\hat{u}_k - u_k}{s}\right)^2\right]$$

# %%
@tag.WilsonCowan
@ObjectiveFunction(tags={'lorentz', 'cauchy', 'breit-wigner', 'forward'})
def WC_lorentz(model, k, s=1):
    "Log Lorentz objective"
    u_predict = model.u_upd(model, k); u_k = model.u[k]
    return -shim.log(1+((u_predict-u_k)/s)**2).sum()
