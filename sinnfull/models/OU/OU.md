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

# Ornstein-Uhlenbeck

```{code-cell} ipython3
:tags: [remove-input]

from __future__ import annotations
```

```{code-cell} ipython3
:tags: [remove-input]

import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')
```

```{code-cell} ipython3
:tags: [hide-input]

from typing import Any, Optional, Union
import numpy as np
import pymc3 as pm
from pydantic import validator
import theano_shim as shim
from mackelab_toolbox.typing import (
    Array, FloatX, Shared, Tensor, AnyRNG, RNGenerator)

from sinn.models import  ModelParams, updatefunction, initializer
from sinn.histories import TimeAxis, Series, AutoHist
from sinn.utils import unlocked_hists

from sinnfull.utils import add_to
from sinnfull.models.base import Model, Param, tag
from sinnfull.typing_ import IndexableNamespace
```

```{code-cell} ipython3
from sinn.histories import Spiketrain
```

```{code-cell} ipython3
Spiketrain.__hash__
```

```{code-cell} ipython3
__all__ = ['OU_AR', 'OU_finite_noise']
```

:::{Note}
The formulations of the OU process below are written explicitely as functions of $\log \tilde{τ}$ and $\log \tilde{σ}$, where $\tilde{τ}$ and $\tilde{σ}$ are the parameters controlling the correlation time and noise strength of the process respectively.

We do this because these are scale parameters that may vary over many orders of magnitude, and we've found that inferrence works best for such parameters when performed in log space. Moreover, since both are strictly positive, this transformation is well-defined and bijective.
:::
$\newcommand{\T}{\intercal}$

+++

## AR(1) form

We write the Ornstein-Uhlenbeck process in a form that most closely resembles an AR(1) process, where the length of a time step implicitely defines the strength of the noise.

In these equations, the sign and magnitude of the mixing matrix are separated into $\tilde{A}$ and $\tilde{W}$, such that the elements of $\tilde{A}$ are $\pm 1$ and those of $\tilde{W}$ are strictly positive.

\begin{align}
d\tilde{I}^m &= -\frac{(\tilde{I}^m - \tilde{\mu}^m)}{\exp (\log \tilde{\tau}^m)} \, dt \,+\, \exp(\log \tilde{\sigma}^m) \sqrt{2} \, dB^m \,,\quad m=1,\dotsc,\tilde{M}\,; \\
I^j &= \sum_{m=1}^{\tilde{M}} \tilde{A}_m^j \tilde{W}_m^{j} \tilde{I}^m \,.
\end{align}

+++

:::{Note}

This parameterization is Markovian: the distribution of $I_k$ depends only on $I_{k-1}$. This allows the likelihood to factorize over time points, but may make it more difficult to infer $\tilde{τ}$ or $\tilde{σ}$.
:::

+++

### Update equation
\begin{align}
\tilde{I}_k &\sim \mathcal{N}\biggl( \tilde{I}_{k-1} - \frac{\tilde{I}_{k-1} - \tilde{\mu}}{\exp(\log \tilde{\tau})} \Delta t,\; 2 \exp(\log \tilde{\sigma})^2 \Delta t \biggr) \\
I_k &= \tilde{W} \tilde{I}_k
\end{align}

:::{margin} Code
`OU_AR`: Parameters
`OU_AR`: Dynamical equations
:::

```{code-cell} ipython3
:tags: [hide-input]

class OU_AR(Model):
    time: TimeAxis

    ## Parameters ##
    class Parameters(ModelParams):
        μtilde   : Shared[FloatX,1]
        logτtilde: Shared[FloatX,1]
        logσtilde: Shared[FloatX,1]
        Wtilde   : Shared[FloatX,2]
        Atilde   : Param[np.int16,2]
        M        : Union[int,Array[np.integer,0]]
        Mtilde   : Union[int,Array[np.integer,0]]
        @property
        def τtilde(self):
            return shim.exp(self.logτtilde)
        @property
        def σtilde(self):
            return shim.exp(self.logσtilde)
        @validator('M', 'Mtilde')
        def only_plain_ints(cls, v):
            return int(v)
    params: Parameters

    ## Other variables retrievable from `self` ##
    I     : Series=None
    Itilde: Series=None
    rng   : AnyRNG=None

    ## State = the histories required to make dynamics Markovian
    class State:
        Itilde: Any  # The type is unimportant, so use Any

    ## Initialization ##
    # Allocate arrays for dynamic variables, and add padding for initial conditions
    # NB: @initializer methods are only used as defaults: if an argument is
    #     explicitely given to the model, its method is NOT executed at all.
    @initializer('Itilde')
    def Itilde_init(cls, Itilde, time, Mtilde):
        return Series(name="Itilde", time=time, shape=(Mtilde,),
                      dtype=shim.config.floatX)
    @initializer('I')
    def I_init(cls, I, time, M):
        return Series(name="I", time=time, shape=(M,),
                      dtype=shim.config.floatX, iterative=False)
    @initializer('rng')
    def rng_init(cls, rng):
        """Note: Instead of relying on this method, prefer passing `rng` as an
        argument, so that all model components have the same RNG."""
        return shim.config.RandomStream()

    def initialize(self, initializer=None):
        """Set the initial conditions."""
        # Skip any already initialized histories; in this model there is only
        # the Itilde history.
        if self.Itilde.cur_tidx >= 0:
            return
        self.Itilde.pad(1)
        if initializer == 'stationary':
            # TODO: Allow setting the rng, without it being symbolic.
            #       Maybe use the one underlying self.rng ?
            stats = self.stationary_stats_eval()
            Itilde_init = self.rng.normal(loc=stats['mean'],
                                          scale=stats['std'],
                                          size=(self.M,))
            self.Itilde[-1] = shim.eval(Itilde_init)
        elif isinstance(initializer, dict):
            for k, v in initializer.values:
                h = getattr(self,k,None)
                if h is not None:
                    assert isinstance(h, History)
                    assert h.pad_left >= 1
                    h[-1] = initializer[k]
        else:
            self.Itilde[-1] = 0

    ## Dynamical equations ##
    @updatefunction('Itilde', inputs=['Itilde', 'rng'])
    def Itilde_upd(self, tidx):
        μ=self.μtilde; τ=self.τtilde; σ=self.σtilde;
        Mtilde=self.Mtilde; dt=self.dt
        Itilde=self.Itilde
        dt = getattr(dt, 'magnitude', dt)  # In case 'dt' is a Pint or Quantities object
        avg = Itilde(tidx-1) - (Itilde(tidx-1)-μ)/τ * dt
        std = σ*shim.sqrt(2*dt)
        return self.rng.normal(avg=avg, std=std, size=(Mtilde,))
    @updatefunction('I', inputs=['Itilde'])
    def I_upd(self, tidx):
        Wtilde=self.Wtilde; Atilde=self.Atilde; Itilde = self.Itilde
        return shim.dot(Atilde*Wtilde, Itilde(tidx))
```

### Stationary distributions

\begin{align}
\mathbb{E}(\tilde{I}_\infty) &= \tilde{\mu} & \mathbb{E}(I_\infty) &= (\tilde{A} \odot \tilde{W}) \tilde{\mu} \,; \\
\mathrm{Var}(\tilde{I}_\infty) &= \tilde{\sigma}^2 \tau & \mathrm{Cov}(I_\infty) &= (\tilde{A} \odot \tilde{W}) \, \tilde{\sigma}^2 \tau \, (\tilde{A} \odot \tilde{W})^\T \,.\\
\end{align}

Here $\odot$ denotes the Hadamard product.

:::{margin} Code
`OU_AR`: Stationary distribution
:::

```{code-cell} ipython3
:tags: [hide-input]

    @add_to('OU_AR')
    @classmethod
    def _stationary_stats(cls, params: ModelParams):
        return {'Itilde': {'mean': params.μtilde, 'std': params.σtilde*shim.sqrt(params.τtilde)}}

    @add_to('OU_AR')
    @classmethod
    def stationary_dist(cls, params: ModelParams, seed=None):
        """
        Return a PyMC3 model corresponding to the process' stationary distribution
        with the current model parameters.
        """
        # Variable names must match those of the histories
        stats = cls.stationary_stats(params)
        Wtilde = params.Wtilde; A = params.A
        with pm.Model() as statdist:
            Itilde_stationary = pm.Normal(
                "Itilde",
                mu=stats['Itilde']['mean'], sigma=stats['Itilde']['std'],
                shape=(params.Mtilde,)
            )
            I_stationary = pm.Deterministic(
                "I", shim.dot(A*Wtilde, Itilde_stationary))
                # FIXME?: Repetition with I_upd
        return statdist
```

### Variables

|**Model variable**| Identifier | Type | Description |
|--|--|--|--
|$I$| `I` | dynamic variable | Output OU process |
|$\tilde{I}$| `Itilde` | dynamic variable | Internal independent OU process |
|$\tilde{\mu}$| `μtilde` | parameter | mean of $\tilde{I}$ |
|$\log \tilde{\tau}$| `logτtilde` | parameter | (log of the) correlation time scale |
|$\log \tilde{\sigma}$| `logσtilde` | parameter | (log of the) noise strength |
|$\tilde{W}$ | `Wtilde` | parameter | mixing of independent OU components (magnitude) |
|$\tilde{A}$ | `Atilde` | parameter | mixing of independent OU components (sign) |
|$M$ | `M` | parameter | number of output components (dim $I$) |
|$\tilde{M}$ | `Mtilde`| parameter | number of independent OU processes (dim $\tilde{I}$) |

+++

### Identifiability of $\tilde{W}$

The mixing matrix $\tilde{W}$ is degenerate in two ways:

1. by rescaling the rows $m$ and the noise amplitude $σ^m$;
2. by permuting the components $m$ of $\tilde{W}$ and $\tilde{I}$.

We therefore add the following restrictions to make it identfiable:

1. We require that columns of $\tilde{W}$ sum to one. This allows us to interpret $\tilde{σ}^m$ as the strength of the noise contributions coming from $\tilde{I}^m$.
2. We order the components of $\tilde{I}$ by their noise strength $\tilde{σ}^m$.

Specifically, we do the following:

1. *Scaling:*
   We scale the sum of each column of $\tilde{W}$ to ±1 and collect that factor in $(\tilde{σ}^m)^2$. Denoting by a prime (${}'$) the quantities before this rescaling, we have
   \begin{align}
     \tilde{I} &= \tilde{σ}\odot\tilde{I}' \,,&
     \tilde{μ} &= \tilde{σ}\odot\tilde{μ}' \,, &
     \tilde{W} &= \frac{1}{\tilde{σ}}\odot \tilde{W}' \,.
   \end{align}
    (One easily verifies that these transformations leave the moments of the differentials of $I$ unchanged.)

2. *Permutation:*
   The degeneracy via permutations introduces multiple modes in the likelihood, but because these are disconnected, it does not impact convergence (although each fit may converge to a different mode). So we can ignore it during the fit, and remove it afterwards by identifying the modes (in mathematical terms, quotienting out the group of permutations). In practice, we order the components in descending order by the scaled out $σ^m$ computed in the previous step. Note that this choice satisfies the condition of smoothness.
   \begin{equation}
     \tilde{σ}^1 \geq \tilde{σ}^2 \geq \dotsb \geq \tilde{σ}^{\tilde{M}}
   \end{equation}
   Note that the likelihood remains smooth in the parameters despite the additional permutation.

:::{margin} Code
`OU_AR`: Degenaracy removal
:::

```{code-cell} ipython3
:tags: [hide-input]

    @add_to('OU_AR')
    def remove_degeneracies(self, params: Optional[OUInput.Parameters]=None,
                            exclude: List[Union[History,str]]=()
                           ) -> Optional[OUInput.Parameters]:
        """
        Parameters
        ----------
        params:
            None: Both parameters and histories are updated in place.
            [Parameters]: A new, non-degenerate parameter set is created and returned.
                The model's own parameters and histories are left untouched.

        exclude: list of histories
            Ignored if params != None.
            When updating in-place, these histories are left untouched; this is useful
            when we want to treat some latent histories has observed.
            Histories can be specified explicitely or by name.
            Extra entries are ignored, so it is safe to pass `optimizer.observed_hists`.
        """
        # Allow this to work as a class method
        if isinstance(self, (ModelParams, IndexableNamespace)):
            assert params is None
            params = self
            self = None  # Only need 'self' when 'inplace' is True
            in_place = False
        else:
            in_place = (params is None)

        # Grab parameter (Θ) and history values
        if in_place:
            params = self.params
        Θ = params.get_values() if hasattr(params, 'get_values') else params
        if in_place:
            #I = self.I.get_trace(include_padding=True)
            Itilde = self.Itilde.get_trace(include_padding=True)

        # Remove scaling degeneracy
        σfactor = abs(Θ.Wtilde.sum(axis=0))
        Θ.logσtilde += np.log(σfactor)  # Multiply by factor = add log of factor to log
        Θ.μtilde *= σfactor
        Θ.Wtilde /= σfactor
        if in_place:
            Itilde   *= σfactor

        # Remove permutation degeneracy
        assert Θ.Mtilde % 2 == 0
        blocks = [np.s_[:Θ.Mtilde//2], np.s_[Θ.Mtilde//2:]]  # Note: must be slices, so that we have views
        for bslc in blocks:
            σsort = np.argsort(Θ.logσtilde[bslc])[::-1]
            Θ.logσtilde[bslc] = Θ.logσtilde[bslc][σsort]
            Θ.μtilde[bslc] = Θ.μtilde[bslc][σsort]
            Θ.logτtilde[bslc] = Θ.logτtilde[bslc][σsort]
            Θ.Wtilde[:,bslc] = Θ.Wtilde[:,bslc][:,σsort]
            Θ.Atilde[:,bslc] = Θ.Atilde[:,bslc][:,σsort]
            if in_place:
                # Only Itilde changes, not I
                Itilde[..., bslc] = Itilde[..., bslc][..., σsort]

        # Update histories
        if in_place:
            # Only Itilde changes, not I
            if self.Itilde not in exclude and self.Itilde.name not in exclude:
                with unlocked_hists(self.Itilde):
                    self.Itilde[:self.Itilde.cur_tidx+1] = Itilde[:]

        # Return
        if in_place:
            self.params.set_values(Θ)
        else:
            return Θ
```

### Testing

+++

It can be useful to also define a `get_test_parameters` method on the class, to run tests without needing to specify a prior.

:::{margin} Code
`OU_AR`: Test parameters
:::

```{code-cell} ipython3
:tags: [hide-input]

    @add_to('OU_AR')
    @classmethod
    def get_test_parameters(cls, rng: Optional[RNGenerator]=None):
        """
        :param:rng: Any value accepted by `numpy.random.default_rng`.
        """
        rng = np.random.default_rng(rng)
        M = rng.integers(1,5)
        Mtilde = rng.integers(1,3)*2
        Θ = cls.Parameters(
            μtilde = rng.normal(size=(Mtilde,)),
            logτtilde = rng.normal(size=(Mtilde,)),
            logσtilde = rng.normal(size=(Mtilde,)),
            Wtilde = abs(rng.normal(size=(M, Mtilde))),  # ~ Halfnormal
            M      = M,
            Mtilde = Mtilde,
            A      = np.concatenate((np.ones(Mtilde//2, dtype=np.int16),
                                     -np.ones(Mtilde-Mtilde//2, dtype=np.int16)))
        )
        return cls.remove_degeneracies(Θ)
```

```{code-cell} ipython3
:tags: [remove-input]

OU_AR.update_forward_refs()  # Always safe to do this & sometimes required
```

The following test integrates the `OU_AR` model, and checks that $I$ is unchanged by the call to `remove_degeneracies`.

```{code-cell} ipython3
:tags: [hide-input]

## Test OUInput.remove_degeneracies ##
if __name__ == "__main__":
    time = TimeAxis(min=0, max=1, step=2**-5, unit=sinnfull.ureg.s)
    Θ_OU = OU_AR.Parameters(
        μtilde=np.float64([2, 0]),
        logτtilde=np.float64([1, 1]), logσtilde=np.float64([0.08, 1.1]),
        Atilde=np.int8([[1, -1],[1, -1]]), Wtilde=np.float64([[2, 3], [1.6, 2.2]]),
        M=2, Mtilde=2
    )
    model = OU_AR(time=time, params=Θ_OU, rng=shim.config.RandomStream())
    model.integrate('end', histories='all')

    orig_I = model.I.get_trace().copy()
    orig_Itilde = model.Itilde.get_trace().copy()

    # I = (Atilde * Wtilde) @ Itilde
    assert np.all(orig_I == np.array([(model.Atilde*model.Wtilde) @ It
                                      for It in orig_Itilde]))

    orig_logσtilde = Θ_OU.logσtilde.copy()
    assert np.all(model.logσtilde == Θ_OU.logσtilde)

    model.remove_degeneracies()

    # Parameter values were changed in-place
    assert np.all(model.logσtilde == Θ_OU.logσtilde)
    assert np.all(model.logσtilde != orig_logσtilde)

    # Itilde has changed. I has not
    assert np.all(orig_Itilde != model.Itilde.get_trace())
    assert np.all(orig_I == model.I.get_trace())

    # We still have I = Wtilde @ Itilde
    assert np.all(np.isclose(orig_I,
                             np.array([(model.Atilde*model.Wtilde) @ It
                                       for It in model.Itilde.get_trace()])))
```

```{code-cell} ipython3
:tags: [hide-cell]

if __name__ == "__main__":
    # Use T >= 50
    model.Itilde.data.mean(axis=0)
    model.Itilde.data.std(axis=0)
    model.stationary_stats()['Itilde']
```

## Finite noise form

Another way of expressing an OU model is the following:
\begin{align}
d\tilde{I}^m &= -\frac{(\tilde{I}^m - \tilde{\mu}^m)}{\tilde{\tau}^m} \, dt \,+\, \tilde{\sigma}^m \sqrt{\frac{2}{\tilde{\tau}^m}} \, dB^m \,,\quad m=1,\dotsc,\tilde{M}\,; \\
I^j &= \sum_{m=1}^{\tilde{M}} \tilde{W}_m^{j} \tilde{I}^m \,,
\end{align}

The main advantage of this form is that the stationary distribution is now independent of the correlation time $\tilde{τ}$, and therefore always has finite noise amplitude. This makes it appropriate for taking the white noise (zero-correlation) limit.

+++

### Update equation
\begin{align}
\tilde{I}_k &\sim \mathcal{N}\biggl( \tilde{I}_{k-1} - \frac{\tilde{I}_{k-1} - \tilde{\mu}}{\tilde{\tau}} \Delta t,\; \tilde{\sigma}^2 {\frac{2 \Delta t}{\tilde{\tau}}} \biggr) \\
I_k &= \tilde{W} \tilde{I}_k
\end{align}

+++

:::{note}

It is possible to avoid redundant definitions by inheriting another model. However rather than inheriting directly, one must do so via a mixin class, as we do below for `OU_FiniteNoise`.
:::

:::{margin} Code
`OU_FiniteNoise`: Dynamical equations
:::

```{code-cell} ipython3
:tags: [hide-input]

class OU_FNMixin:
    # Note that the mixin class does NOT inherit from Model

    @updatefunction('Itilde', inputs=['Itilde', 'rng'])
    def Itilde_upd(self, tidx):
        μ=self.μtilde; τ=self.τtilde; σ=self.σtilde;
        Mtilde=self.Mtilde; dt=self.dt
        Itilde=self.Itilde
        dt = getattr(dt, 'magnitude', dt)  # In case 'dt' is a Pint or Quantities
        avg = Itilde(tidx-1) - (Itilde(tidx-1)-μ)/τ * dt
        std = σ*shim.sqrt(2*dt/τ)  # <<< Only difference with OUInput: divide by τ
        return self.rng.normal(avg=avg, std=std, size=(Mtilde,))
```

### Stationary state
\begin{align}
\mathbb{E}(\tilde{I}_\infty) &= \tilde{\mu} & \mathbb{E}(I_\infty) &= \tilde{A} \tilde{W} \tilde{\mu} \,; \\
\mathrm{Var}(\tilde{I}_\infty) &= \tilde{\sigma}^2 & \mathrm{Cov}(I_\infty) &= \tilde{A} \tilde{W} \, \tilde{\sigma}^2 \, \tilde{W}^\T \tilde{A}^\T \,.\\
\end{align}

:::{margin} Code
`OU_FiniteNoise`: Stationary distribution
:::

```{code-cell} ipython3
:tags: [hide-input]

    @add_to('OU_FNMixin')
    @classmethod
    def _stationary_stats(cls, params: ModelParams):
        return {'Itilde': {'mean': params.μtilde, 'std': params.σtilde}}
```

```{code-cell} ipython3
:tags: [hide-input]

class OU_FiniteNoise(OU_FNMixin, OU_AR):
    pass
```

```{code-cell} ipython3
:tags: [remove-input]

OU_FiniteNoise.update_forward_refs()  # Always safe to do this & sometimes required
```
