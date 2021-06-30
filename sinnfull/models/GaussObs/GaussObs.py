# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,md:myst
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
# # Gaussian observation model
# $
# \newcommand{\obs}{\scriptscriptstyle\mathrm{obs}}
# \newcommand{\dyn}{\scriptscriptstyle\mathrm{dyn}}
# $

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('theano')

# %% tags=["hide-cell"]
from typing import Any, Optional, Union
import numpy as np
import theano_shim as shim
from mackelab_toolbox.typing import (
    FloatX, Shared, Array, AnyRNG, RNGenerator)
from sinn.models import ModelParams, updatefunction, initializer
from sinn.histories import TimeAxis, Series
from sinn.utils import unlocked_hists

from sinnfull.utils import add_to, add_property_to
from sinnfull.rng import draw_model_sample
from sinnfull.models.base import Model, Param

# %%
__all__ = ['GaussObs']


# %% [markdown]
# ## Uncorrelated Gaussian observation model
#
# We define the *Gaussian observation model* as
#
# :::{math}
# :label: eq:gauss-obs
# \begin{aligned}
# \bar{u}_k \sim \mathcal{N}\left(\bar{W} u_k, \bar{Σ}\right) \,.
# \end{aligned}
# :::
#
# This is simultaneously the definition of the model and of its likelihood.
#
# :::{admonition} Notation  
# As a convention, we denote variables and parameters of an observation model with a bar, those of an input model with a tilde, and those of a dynamical model without additional symbols.
#
# Furthermore, we refer to the dimensions of the output space as *channels* (in analogy with the channels of a measurement device) and denote their number with $C$.  
# :::

# %% [markdown]
# An observation model serves to link a dynamical model to what is actually observed. It typically has the following properties:
#
# - no dependence on past time points;
# - conditional independence of the observations (i.e. noise is added independently to each observed component).
#
# These are not strict requirements, but they help make the inference problem less tractable. Moreover, such interactions are usually better captured by the dynamical or input models.
#
# We made the choice here to formulate Eq. [$u_k$](eq.gauss-obs) directly in discretized time. This is most appropriate if we believe $Σ^{\obs}$ is independent of the time step. In some cases[^1] parameterizing in terms of a continuous process and explicitely carrying out a discretization may lead to inferred parameters which are less sensitive to the time step.
#
# The matrix $W^{\obs}$ offers the possibility for the observation model to also mix components of the dynamical model. However it is not recommended to leave $W^{\obs}$ unconstrained, as this can make the inference problem much more difficult. Constraints are imposed by selecting an appropriate prior on $W^{\obs}$; a few variants (including $W = I$ and a rectangular $W$) are [already provided](./gaussobs_priors).
#
# Similarly, in many situations it is reasonable to assume that $\bar{Σ}$ is diagonal, which simplifies the inference problem by decorrelating parameters. (“Decorrelated” in the sense that changes to one parameter component will not change the likelihood of other components.) Another benefit to assuming independent noise is that elementwise functions are computationally more efficient and better supported by automatic differentiation.
#
# [^1]: For example, if the noise is tied to a measurement device which integrates for one time step. In that case, longer time steps = longer integration time = smaller variability; thus the discrete time variance strongly depends on the time step, even if the continuous time variance does not.

# %% [markdown]
# ### Variables
#
# |**Model variable**| Identifier | Type | Description |
# |--|--|--|--
# |${u} \in \mathbb{R}^M$| `u` | dynamic variable | input; usu. from a dynamical model |
# |$\bar{u}  \in \mathbb{R}^C$| `ubar` | dynamic variable | observed quantities |
# |$\bar{W}$| `Wbar` | parameter | linear projection dynamic to observed variables |
# |$\bar{Σ} \in \mathbb{R}_+^C$| `Σbar` | parameter | variance of the observation noise |
# |$M$ | `M` | parameter | no. of dynamic components |
# |$C$ | `C` | parameter | number of observed channels |
#
# :::{remark}  
# $\bar{Σ}$ is a vector (not a matrix). This reflects the assumption of independent noise, and simplifies differentiation.  
# :::

# %%
class GaussObs(Model):
    time: TimeAxis
        
    class Parameters(ModelParams):
        C:  Union[int,Array[np.integer,0]]
        M:  Union[int,Array[np.integer,0]]
        Wbar: Param[FloatX, 2]
        Σbar: Shared[FloatX, 1]
            
    u   : Series
    ubar: Series=None
    rng : AnyRNG=None
    
    class State:
        pass  # No dependence on t-1
    
    @initializer('ubar')
    def ubar_init(cls, ubar, time, C):
        return Series(name='ubar', time=time, shape=(C,), dtype=shim.config.floatX)
    @initializer('rng')
    def rng_init(cls, rng):
        """Note: Prefer passing as argument, so that all model components have the same RNG."""
        return shim.config.RandomStream()
    
    def initialize(self, initializer=None):
        pass
    
    @updatefunction('ubar', inputs=['u', 'rng'])
    def ubar_upd(self, tidx):
        Wbar=self.Wbar; Σbar=self.Σbar; M=self.M
        u=self.u; rng=self.rng
        return shim.dot(Wbar, u(tidx)) \
            + rng.normal(avg=0, std=shim.sqrt(Σbar), size=(M,))


# %% [markdown]
# :::{margin} Code `GaussObs`
# Test parameters
# :::

    # %%
    @add_to('GaussObs')
    @classmethod
    def get_test_parameters(cls, rng: Union[None,int,RNGenerator]=None):
        rng = np.random.default_rng(rng)
        #M = rng.integers(1,5)
        M = 2  # Currently no way to ensure submodels draw the same M
        C = M
        Θ = cls.Parameters(
            M = M,
            C = M,
            Wbar = np.eye(C),
            Σbar = rng.uniform(0, 2, size=M)
        )
        return Θ

# %%
GaussObs.update_forward_refs()

# %% [markdown]
# ## Examples

# %% [markdown]
# Gaussian observations of a sinusoid

# %%
if __name__ == "__main__":
    from sinn.histories import HistoryUpdateFunction
    import holoviews as hv
    hv.extension('bokeh')
    
    Θ = GaussObs.get_test_parameters()
    
    time  = TimeAxis(min=0, max=6, step=2**-5)
    u = Series(name='u', time=time, shape=(Θ.M,), dtype=shim.config.floatX)

    # %%
    obs_model = GaussObs(time=time, u=u, params = Θ)

    # %%
    def u_upd(model, k):
        return np.sin(np.array([0,np.pi/2]) + model.time[k])
    u.update_function = HistoryUpdateFunction(
        namespace=obs_model, inputs=(), func=u_upd)

    # %%
    obs_model.integrate(upto='end', histories='all')

    # %%
    u_curves = [hv.Curve(trace, label=f"u_{i}")
                for i, trace in enumerate(u.traces)]
    ubar_curves = [hv.Curve(trace, label=f"ubar_{i}")
                   for i, trace in enumerate(obs_model.ubar.traces)]
    for c in u_curves:
        c.opts(line_dash='solid')
    for c in ubar_curves:
        c.opts(alpha=0.7)
    hv.Overlay(ubar_curves + u_curves) \
        .opts(width=500, legend_position='right')

# %%
