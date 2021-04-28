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
# # Wilson-Cowan

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')

# %%
# %debug

# %% tags=["hide-cell"]
from typing import Any, Optional, Union
import numpy as np
import theano_shim as shim
from mackelab_toolbox.typing import FloatX, Shared
from sinn.models import ModelParams, updatefunction, initializer
from sinn.histories import TimeAxis, Series, AutoHist
from sinn.utils import unlocked_hists

from sinnfull.utils import add_to, add_property_to
from sinnfull.models.base import Model, Param

# %%
__all__ = ['WilsonCowan']


# %% [markdown]
# ## Wilson-Cowan dynamical model
#
# $\newcommand{\tag}[1]{\qquad\text{(#1)}}$
# We take the following Wilson-Cowan model
# :::{math}
# :label: eq:wc-def  
# \begin{align}
# α_e^{-1} \frac{d}{dt}{u}^e &= L[{u}^e] + {w}_e^{e} F^e[{u}^e] + {w}_i^{e} F^i[{u}^i] + I^e(t) \,,\\
# α_i^{-1} \frac{d}{dt}{u}^i &= L[{u}^i] + {w}_e^{i} F^e[{u}^e] + {w}_i^{i} F^i[{u}^i] + I^i(t) \,;
# \end{align}  
# :::
# where
# \begin{align}
# F_j[u] &= (1 + \exp[-β^j(u - h^j)])^{-1} \,, \\
# L[{u}] &= -{u} \,.
# \end{align}
#
# In the following it will be easier to work with the vectorized form of this equation, i.e.
#
# \begin{equation}
# α^{-1} \frac{d}{dt}{u}_t = L[{u}_t] + {w} F[{u}_t] + I_t \,.
# \end{equation}

# %% [markdown]
# ### Update equation
# Discretized with an Euler scheme, this leads to the following update equations:
# \begin{align}
# {u}_k &= {u}_{k-1} + (α\,Δt) \odot \bigl(L[{u}_{k-1}] + {w} F[{u}_{k-1}] + I_{k}\bigr) \,, \\
# {u}_k &\in \mathbb{R}^M \,,
# \end{align}
# where $\odot$ denotes the Hadamard product.
#
# :::{margin} Code
# `WilsonCowan`: Parameters  
# `WilsonCowan`: Dynamical equations
# :::

# %% tags=["hide-input"]
class WilsonCowan(Model):
    time :TimeAxis

    class Parameters(ModelParams):
        α :Shared[FloatX,1]
        β :Shared[FloatX,1]
        w :Shared[FloatX,2]
        h :Shared[FloatX,1]
        M :int
    params :Parameters

    ## Other variables retrievable from `self` ##
    u :Series=None
    I :Series

    # State = the histories required to make dynamics Markovian
    class State:
        u :Any
        I :Any

    ## Initialization ##
    # Allocate arrays for dynamic variables, and add padding for initial conditions
    # NB: @initializer methods are only used as defaults: if an argument is
    #     explicitely given to the model, its method is NOT executed at all.
    @initializer('u')
    def u_init(cls, u, time, M):
        return Series(name='u', time=time, shape=(M,), dtype=shim.config.floatX)

    def initialize(self, initializer=None):
        self.u.pad(1)

    ## Dynamical equations ##
    def L(self, u):
        return -u
    def F(self, u):
        β=self.β; h=self.h
        return (1 + shim.exp(-β*(u-h)))**(-1)
    @updatefunction('u', inputs=['u', 'I'])
    def u_upd(self, tidx):
        α=self.α; β=self.β; w=self.w; h=self.h; dt=self.dt
        L=self.L; F=self.F
        u=self.u; I=self.I
        return u(tidx-1) + α*dt * (L(u(tidx-1)) + shim.dot(w, (F(u(tidx-1)))) + I(tidx))


# %% [markdown]
# ### Analytics and stationary state
#
# Eq. [du/dt](eq:wc-def) decomposes as a deterministic transformation of ${u}_t$ and the addition of a random variable $I_t$; thus if $I_t$ is Gaussian, ${u}$ also follows a Gaussian process, and it suffices to describe its mean and covariance. Expanding $\langle {u}_{t+dt} \rangle$ and $\langle {u}_{t+dt}^2 \rangle$, and assuming additive noise, we can formally write the differential equations for the bare moments:
#
# \begin{align}
# d \langle u_t \rangle \overset{\scriptstyle{\text{def}}}{=} {\langle {u}_{t+dt} \rangle - \langle {u}_{t} \rangle} &= \langle L[{u}_t] \rangle \,dt + {w} \langle F[{u}_t] \rangle \,dt + \langle d I_t \rangle \\
# d \langle u_t^2 \rangle \overset{\scriptstyle{\text{def}}}{=} {\langle {u}_{t+dt}^2 \rangle - \langle {u}_{t}^2 \rangle} &= 2 \langle {u}_t L[{u}_t] \rangle \,dt + 2 {w} \langle u_t F[{u}_t] \rangle \,dt + 2 \langle {u}_t \rangle \langle d I_t \rangle + 2 \langle I_t \rangle \langle dI_t\rangle + \langle (d I_t)^2 \rangle
# \end{align}
# (Here we used the equalities $\langle {u}_t d I_t \rangle = \langle {u}_t \rangle \langle d I_t \rangle$ and $\langle {I}_t d I_t \rangle = \langle {I}_t \rangle \langle d I_t \rangle$, which are always true for additive noise, and true in general for an Itô SDE.)
#
# If $I_t$ is e.g. a white noise or OU process, we can obtain expressions for $\langle dI_t \rangle$ and $\langle (d I_t)^2 \rangle$ fairly easily.
#
# Moreover, under those conditions these equations are also closed, since then $u$ is also a Gaussian process: the expectations involving $u_t$ are taken with respect to the Gaussian with mean variance given by the solutions. In some cases (such as the proposed linear $L[\cdot]$) these may be carried out analytically, otherwise we may perform a Taylor expansion and take the expectation of the polynomials; the proposed sigmoid for $F[\cdot]$ is antisymmetric and linear around $h$, and should be tractable with this approach.

# %% [markdown]
# ### Variables
# |**Model variable**| Identifier | Type | Description |
# |--|--|--|--
# |${u}$| `u` | dynamic variable | population activity |
# |$I$| `I` | dynamic variable | external input |
# |$α$| `α` | parameter | time scale |
# |$β$| `β` | parameter | sigmoid steepnees |
# |$h$| `h` | parameter | sigmoid centre |
# |${w}$| `w` | parameter | connectivity |
# |$M\in 2\mathbb{N}$| `M` | parameter | number of populations; even because populations are split into E/I pairs |

# %% tags=["remove-cell"]
WilsonCowan.update_forward_refs()

# %% tags=["remove-cell"]
if __name__ == "__main__":
    from sinnfull.models import models, priors, paramsets, objectives

    # %%
    models

    # %%
    priors

    # %%
    objectives

# %%
