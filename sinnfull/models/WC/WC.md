---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: py:percent,md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
kernelspec:
  display_name: Python (sinn-full)
  language: python
  name: sinn-full
---

# Wilson-Cowan

```{code-cell}
:tags: [remove-cell]

from __future__ import annotations
```

```{code-cell}
:tags: [remove-cell]

import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')
```

```{code-cell}
:tags: [hide-cell]

from typing import Any, Optional, Union
import numpy as np
import theano_shim as shim
from mackelab_toolbox.typing import FloatX, Shared, Array, RNGenerator
from sinn.models import ModelParams, updatefunction, initializer
from sinn.histories import TimeAxis, Series, AutoHist
from sinn.utils import unlocked_hists

from sinnfull.utils import add_to, add_property_to
from sinnfull.rng import draw_model_sample
from sinnfull.models.base import Model, Param
```

```{code-cell}
__all__ = ['WilsonCowan']
```

## Wilson-Cowan dynamical model

$\newcommand{\tag}[1]{\qquad\text{(#1)}}$
The following form of the Wilson-Cowan model is based on the one used by [Rich et al. (2020)](https://www.nature.com/articles/s41598-020-72335-6).

:::{math}
:label: eq:wc-def
\begin{aligned}
α_e^{-1} \frac{d}{dt}{u}^e &= L[{u}^e] + {w}_e^{e} F^e[{u}^e] + {w}_i^{e} F^i[{u}^i] + I^e(t) \,,\\
α_i^{-1} \frac{d}{dt}{u}^i &= L[{u}^i] + {w}_e^{i} F^e[{u}^e] + {w}_i^{i} F^i[{u}^i] + I^i(t) \,;
\end{aligned}
:::

where
\begin{align}
F_j[u] &= (1 + \exp[-β^j(u - h^j)])^{-1} \,, \\
L[{u}] &= -{u} \,.
\end{align}

In the following it will be easier to work with the vectorized form of this equation, i.e.

\begin{equation}
α^{-1} \frac{d}{dt}{u}_t = L[{u}_t] + {w} F[{u}_t] + I_t \,.
\end{equation}

+++

### Update equation
Discretized with an Euler scheme, this leads to the following update equations:
\begin{align}
{u}_k &= {u}_{k-1} + (α\,Δt) \odot \bigl(L[{u}_{k-1}] + {w} F[{u}_{k-1}] + I_{k}\bigr) \,, \\
{u}_k &\in \mathbb{R}^M \,,
\end{align}
where $\odot$ denotes the Hadamard product.

:::{margin} Code
`WilsonCowan`: Parameters
`WilsonCowan`: Dynamical equations
:::

```{code-cell}
:tags: [hide-input]

class WilsonCowan(Model):
    time: TimeAxis

    class Parameters(ModelParams):
        α: Shared[FloatX,1]
        β: Shared[FloatX,1]
        w: Shared[FloatX,2]
        h: Shared[FloatX,1]
        M: Union[int,Array[np.integer,0]]
    params: Parameters

    ## Other variables retrievable from `self` ##
    u: Series=None
    I: Series

    ## State = the histories required to make dynamics Markovian
    class State:
        u: Any
        I: Any

    ## Initialization ##
    # Allocate arrays for dynamic variables, and add padding for initial conditions
    # NB: @initializer methods are only used as defaults: if an argument is
    #     explicitely given to the model, its method is NOT executed at all.
    @initializer('u')
    def u_init(cls, u, time, M):
        return Series(name='u', time=time, shape=(M,), dtype=shim.config.floatX)

    def initialize(self, initializer=None):
        # Skip any already initialized histories; in this model there is only
        # the u history which has initial conditions.
        if self.u.cur_tidx >= 0:
            return
        # Proceed with initialization
        self.u.pad(1)
        # Generic stuff that could go in Model
        if initializer == "stationary":
            stat_dist = self.stationary_dist(self.params)
            ic = draw_model_sample(stat_dist)
        elif isinstance(initializer, tuple):
            stat_dist = self.stationary_dist
            ic = draw_model_sample(stat_dist, key=initializer)
        elif isinstance(initializer, dict):
            ic = initializer
        else:
            ic = None
        if ic:
            for k, v in initializer.items():
                h = getattr(self,k,None)
                if h is not None:
                    assert isinstance(h, History)
                    assert h.pad_left >= 1
                    h[-1] = initializer[k]
        # Non-generic stuff
        else:
            self.u[-1] = 0

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
        dt = getattr(dt, 'magnitude', dt)  # In case 'dt' is a Pint or Quantities
        return u(tidx-1) + α*dt * (L(u(tidx-1)) + shim.dot(w, (F(u(tidx-1)))) + I(tidx))
```

:::{margin} Code `WilsonCowan`
Test parameters
:::

```{code-cell}
:tags: [hide-cell]

    @add_to('WilsonCowan')
    @classmethod
    def get_test_parameters(cls, rng: Union[None,int,RNGenerator]=None):
        rng = np.random.default_rng(rng)
        #M = rng.integers(1,5)
        M = 2  # Currently no way to ensure submodels draw the same M
        A = np.concatenate((np.ones(M//2, dtype='int16'),
                            -np.ones(M//2, dtype='int16')))
        _w_mag = rng.lognormal(-0.5, 3, size=(M,M))
        Θ = cls.Parameters(
            α = rng.lognormal(-2, 3, size=(M,)),
            β = rng.lognormal(1, 2, size=(M,)),
            w = A*_w_mag,
            h = rng.normal(0., 2, size=(M,)),
            M = M
        )
        return Θ
```

### Analytics and stationary state

Eq. [du/dt](eq:wc-def) decomposes as a deterministic transformation of ${u}_t$ and the addition of a random variable $I_t$; thus if $I_t$ is Gaussian, ${u}$ also follows a Gaussian process, and it suffices to describe its mean and covariance. Expanding $\langle {u}_{t+dt} \rangle$ and $\langle {u}_{t+dt}^2 \rangle$, and assuming additive noise, we can formally write the differential equations for the bare moments:

\begin{align}
d \langle u_t \rangle \overset{\scriptstyle{\text{def}}}{=} {\langle {u}_{t+dt} \rangle - \langle {u}_{t} \rangle} &= \langle L[{u}_t] \rangle \,dt + {w} \langle F[{u}_t] \rangle \,dt + \langle d I_t \rangle \\
d \langle u_t^2 \rangle \overset{\scriptstyle{\text{def}}}{=} {\langle {u}_{t+dt}^2 \rangle - \langle {u}_{t}^2 \rangle} &= 2 \langle {u}_t L[{u}_t] \rangle \,dt + 2 {w} \langle u_t F[{u}_t] \rangle \,dt + 2 \langle {u}_t \rangle \langle d I_t \rangle + 2 \langle I_t \rangle \langle dI_t\rangle + \langle (d I_t)^2 \rangle
\end{align}
(Here we used the equalities $\langle {u}_t d I_t \rangle = \langle {u}_t \rangle \langle d I_t \rangle$ and $\langle {I}_t d I_t \rangle = \langle {I}_t \rangle \langle d I_t \rangle$, which are always true for additive noise, and true in general for an Itô SDE.)

If $I_t$ is e.g. a white noise or OU process, we can obtain expressions for $\langle dI_t \rangle$ and $\langle (d I_t)^2 \rangle$ fairly easily.

Moreover, under those conditions these equations are also closed, since then $u$ is also a Gaussian process: the expectations involving $u_t$ are taken with respect to the Gaussian with mean variance given by the solutions. In some cases (such as the proposed linear $L[\cdot]$) these may be carried out analytically, otherwise we may perform a Taylor expansion and take the expectation of the polynomials; the proposed sigmoid for $F[\cdot]$ is antisymmetric and linear around $h$, and should be tractable with this approach.

+++

## Variables
|**Model variable**| Identifier | Type | Description |
|--|--|--|--
|${u}$| `u` | dynamic variable | population activity |
|$I$| `I` | dynamic variable | external input |
|$α$| `α` | parameter | time scale |
|$β$| `β` | parameter | sigmoid steepness |
|$h$| `h` | parameter | sigmoid centre |
|${w}$| `w` | parameter | connectivity |
|$M\in 2\mathbb{N}$| `M` | parameter | number of populations; even because populations are split into E/I pairs |

```{code-cell}
:tags: [remove-cell]

WilsonCowan.update_forward_refs()
```

## Examples

+++

Wilson-Cowan model driven by Gaussian white noise.

```{code-cell}
:tags: [hide-input, remove-output]

if __name__ == "__main__":
    from sinnfull.models import paramsets
    from sinnfull.models.GWN.GWN import GaussianWhiteNoise
    from sinnfull.rng import get_shim_rng
    from IPython.display import display
    import holoviews as hv
    hv.extension('bokeh')

    # Parameters
    rng_sim = get_shim_rng((1,0), exists_ok=True)
        # exists_ok=True allows re-running the cell
    Θ_wc  = paramsets.WC.rich
    Θ_gwn = paramsets.GWN.rich
    #Θ_gwn['μ'] = [500., -.5]
    assert Θ_wc.M == Θ_gwn.M
    time  = TimeAxis(min=0, max=.4, step=2**-10)

    # Model
    noise = GaussianWhiteNoise(
        time  =time,
        params=Θ_gwn,
        rng   =rng_sim
    )
    model = WilsonCowan(
        time  =time,
        params=Θ_wc,
        I     =noise.ξ
    )
    # Set initial conditions
    model.u[-1] = 0
```

```{code-cell}
:tags: [hide-input, remove-output]

    # Integrate
    noise.integrate(upto='end')  # Integrating the noise first allows a batch call
    model.integrate(upto='end')
```

```{code-cell}
:tags: [hide-input]

    traces = []
    for hist in model.history_set:
        traces.extend( [hv.Curve(trace, kdims=['time'],
                                 vdims=[f'{hist.name}{i}'])
                        for i, trace in enumerate(hist.traces)] )

    display(hv.Layout(traces).cols(Θ_wc.M))
```

Functions $F$ and $L$ used for the simulation above. Right panel is enlarged to show the sigmoid.

```{code-cell}
if __name__ == "__main__":
    panels = []
    for u_arr in [np.linspace(-10, 10), np.linspace(-.03, .03)]:
        curve_F = hv.Curve(zip(u_arr, model.F(u_arr-model.h)), kdims=["u"], label="F(u)")
        curve_L = hv.Curve(zip(u_arr, model.L(u_arr)), kdims=["u"], label="L(u)")
        ov = curve_F * curve_L
        panels.append(ov)
    ov.opts(legend_position="top_left")
    # Zoom rectangle – so small we don't really see it
    umin, ymin = curve_F.data.min(axis=0)
    umax, ymax = curve_F.data.max(axis=0)
    rect = hv.Rectangles([(umin, ymin, umax, ymax)]).opts(color="gray", alpha=.5)
    panels[0] = rect * panels[0]
    # Parameters table
    Θtable = hv.Table([["β", str(model.β)], ["h", str(model.h)]],
                      kdims=["name", "value"])
    panels.append(Θtable)
    layout = hv.Layout(panels).opts(shared_axes=False)
    display(layout)
```

```{code-cell}

```
