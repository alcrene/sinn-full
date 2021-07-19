# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # GaussObs objective functions

# %%
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup("numpy")

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
# The Gaussian observation has the form (see Eq. [./GaussObs.py](eq:gauss-obs))
#
# $$\begin{gathered}
# \bar{u}_k \sim \mathcal{N}\left(\bar{W} u_k, \bar{Σ}\right) \,; \\
# \begin{aligned}[t]
# \bar{W} &\in \mathbb{R}^{C\times N} \,, & \bar{Σ} &\in \mathbb{R}_+^C
# \end{aligned}
# \,.\end{gathered}$$

# %% [markdown]
# ## Public interface
# All functions defined in this module are accessible from the collection `sinnfull.models.objectives.GaussObs`.

# %% [markdown]
# ## Log-likelihood
#
# $$\begin{aligned}
# p(\bar{u}_k | \bar{Σ}) &= \log \left[\frac{1}{\sqrt{\bar{Σ}}} \frac{1}{\sqrt{2 π}} \exp\left(-\frac{1}{2}\left(\frac{\bar{u}_k - μ_k}{\sqrt{\bar{Σ}}}\right)^2 \right)\right] \\
# &= -\frac{1}{2}\log Σ -\frac{(\bar{u}_k - u_k)^2}{2\bar{Σ}} + C
# \end{aligned}$$

# %%
@tag.GaussObs
@ObjectiveFunction(tags={'log L'})
def GaussObs_logp(model, k):
    "Log probability of the Gaussian observation model."
    u=model.u; ubar=model.ubar; Σbar=model.Σbar
    norm = -0.5*shim.log(Σbar)
    gauss = - (ubar(k)-u(k))**2 / (2*Σbar)
    return norm.sum() + gauss.sum()


# %% [markdown]
# When the optimizer evaluates the objective, it is not known in general whether `ubar` or `u` are given or need to be computed. Hence we use round brackets instead of square brackets for indexing.

# %% [markdown]
# ## Test
#
# We test here a basic requirement of an optimization, namely that its maximum corresponds to the parameter values used to generate the data.
# To do this, we generate some data for $u$ and $\bar{u}$, then compute the  `GaussObs_logp` (the likelihood) for the observations $\bar{u}$ for the known value of $\bar{Σ}$. This is compared to the likelihood of randomly drawn parameters on the same data.

# %% [markdown]
# :::{remark}  
# With finite data, there will be parameter sets with a higher likelihood than the true parameters. Thus one should not expect the MLE to coincide exactly with the true parameters, is the test below illustrates.  
# :::

# %%
if __name__ == "__main__":
    import pandas as pd
    from sinnfull.models import models, TimeAxis
    from sinnfull.rng import get_np_rng, get_shim_rng
    from sinn.histories import Series
    gaussobs_cls = models.GaussObs

    # %%
    Θrng = get_np_rng((0,1), exists_ok=True)
    shimrng = get_shim_rng((1,0), exists_ok=True)

    # %%
    time = TimeAxis(min=0, max=10000, step=1)
    Θ = gaussobs_cls.get_test_parameters(Θrng)
    u = Series(name='u', time=time, shape=(Θ.M,), dtype=shim.config.floatX)
    obs_model = gaussobs_cls(time=time, params=Θ, rng=shimrng, u=u)

    # %%
    # Only needed when reintegrating the model
    obs_model.u.unlock()
    obs_model.ubar.unlock()
    obs_model.clear()

    # %%
    u[:] = np.array([0,np.pi/2]) + np.sin(time.stops_array)[:,None]
    u.lock()
    obs_model.integrate(upto='end', histories='all')
    obs_model.ubar.lock()
        # Locking prevents `update_params` from resetting histories

    # %%
    #Klst = [200, 400, 1000, 2000, 4000, 7000, 10000]
    N = 400  # Number of parameter sets to sample
    Klst = np.linspace(200, 10000)
    counts_data = []
    for K in Klst:
        slc = slice(0, K)
        # Objective for true parameters
        true_Θ = obs_model.Σbar
        true_logp = GaussObs_logp(obs_model, slc)/K

        # Objective for randomly drawn parameters
        logps_random = {}
        better_than_true = {}
        for i in range(400):
            Θ = obs_model.get_test_parameters(Θrng)
            obs_model.update_params(Θ)
            obs_model.integrate(upto='end', histories='all')
            logp = GaussObs_logp(obs_model, slc)/len(time)
            logps_random[f"Θ_{i}"] = logp
            if logp > true_logp:
                better_than_true[f"Θ_{i}"] = Θ.Σbar
                
        counts_data.append((K, len(better_than_true)))

        #print(K)
        #print("num > true", len(better_than_true))
        #print("max Δ(log Σ)", max((Σbar-true_Θ).max()
        #                          for Σbar in better_than_true.values()))

# %% [markdown]
# (Not shown: the maximum difference with the “better than” parameters in log space of $Σ$ is easily 1 and goes down very slowly with $K$, the number of time points.)

    # %%
    import holoviews as hv
    hv.extension('bokeh')

    # %%
    hv.Curve(counts_data,
             kdims=['K'],
             vdims=['% Θ w/ better log L'])

# %% [markdown]
# :::{note}  
# The values in the figure above are not independent for two reasons:
#
# - The data segment always starts at 0 and goes up to $K$.
# - The same data are used for all evaluations (i.e. we have frozen observation noise).
#
# Given enough repeated realizations, the curve would approach the expectation value, which is smooth and monotone decreasing. The situation illustrated here however is perhaps more representative of a real-world scenario, where the number of realizations is fixed (and often equal to 1).  
# :::
