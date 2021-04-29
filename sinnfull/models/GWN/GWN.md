---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: py:percent,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python (sinn-full)
  language: python
  name: sinn-full
---

# Gaussian white noise

The Gaussian white noise model is mostly useful as an input to other models, but can also be inferred directly to test algorithms.

+++

:::{attention}  
`GaussianWhiteNoise` is a **noise source**, which for practical and technical reasons we distinguish from *models*. One can think of noise sources as models with no dynamics – since each time point is independent, they can be generated with one vectorized call to the random number generator. The extent to which this distinction is useful is still being worked out.

At present noise sources do **not** inherit from `sinnfull.models.Model`, although much of the API is reproduced. This may change in the future.
:::

```{code-cell} ipython3
:tags: [remove-cell]

from __future__ import annotations
```

```{code-cell} ipython3
:tags: [remove-cell]

import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')
```

```{code-cell} ipython3
:tags: [hide-cell]

from typing import Any, Optional, Union
import numpy as np
import theano_shim as shim
from mackelab_toolbox.typing import FloatX, Shared, Array, AnyRNG, RNGenerator
from pydantic import BaseModel, PrivateAttr  # Move with NoiseSource
import sys

from sinn.models import ModelParams, updatefunction, initializer
from sinn.histories import TimeAxis, Series, HistoryUpdateFunction

from sinnfull.utils import add_to, add_property_to
from sinnfull.models.base import Model, Param
```

```{code-cell} ipython3
__all__ = ['GaussianWhiteNoise']
```

## Model equations

This is a model for $M$ independent noise processes given by:

:::{math}
:label: eq:gwn-def
\begin{aligned}
ξ^m(t) &\sim \mathcal{N}(μ^m, {(σ^m)}^2) \\
\langle (ξ^m(t)-μ^m)(ξ^m(t')-μ^m) \rangle &= {(σ^m)}^2 δ(t-t')
\end{aligned}
:::

where $m \in \{1, \dotsc, M\}$ and $μ^m, σ^m, ξ^m(t) \in \mathbb{R}$. In the equations below, as well as the code, we drop the index $m$ and treat $μ$, $σ$ and $ξ$ as arrays to which operations are applied element wise.

:::{Note}
Below, we write the process explicitely as a function of $\log σ$.
We do this because $σ$ is a scale parameters that may vary over many orders of magnitude, and we've found that inferrence works best for such parameters when performed in log space. Moreover, since it is strictly positive, this transformation is well-defined and bijective.
:::

### Discretized form (“update” equations)

Formally, we define $ξ_k$ as

$$ξ_k := \int_{t_k}^{t_k+Δt} ξ(t') dt' \,.$$

For the limit $\lim_{Δt \to 0} ξ_k$ to exist, we must have

$$ξ_k \sim \mathcal{N}(μ, \exp(\log σ)^2/Δt) \,,$$

which is the definition we use in implementations.

:::{margin} Code  
`GaussianWhiteNoise`: Parameters  
`GaussianWhiteNoise`: Update equation  
:::

```{code-cell} ipython3
:tags: [hide-input]

class GaussianWhiteNoise(BaseModel):
    # Currently we just inherit from plain BaseModel
    # Eventually we may define NoiseSource with common functionality
    time: TimeAxis
        
    class Parameters(ModelParams):
        μ   : Shared[FloatX, 1]
        logσ: Shared[FloatX, 1]
        M   : Union[int,Array[np.integer,0]]
    params: GaussianWhiteNoise.Parameters
        
    ξ  : Series=None
    rng: AnyRNG=None
        
    @initializer('ξ')
    def ξ_init(cls, ξ, time, M):
        return Series(name='ξ', time=time, shape=(M,), dtype=shim.config.floatX)
    @initializer('rng')
    def rng_init(cls, rng):
        """Note: Instead of relying on this method, prefer passing `rng` as an
        argument, so that all model components have the same RNG."""
        return shim.config.RandomStream()
        
    @updatefunction('ξ', inputs=['rng'])
    def ξ_upd(self, k):
        μ = self.params.μ
        σ = shim.exp(self.params.logσ)
        M = self.M; dt = self.dt
        dt = getattr(dt, 'magnitude', dt)
        return self.rng.normal(avg=μ, std=σ/shim.sqrt(dt), size=(M,))
    
    ## Stuff that could be in NoiseSource class
    
    _hist_identifiers  : List[str]=PrivateAttr(['ξ'])
    _kernel_identifiers: List[str]=PrivateAttr([])
    _model_identifiers : List[str]=PrivateAttr([])
            
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ξ.update_function = HistoryUpdateFunction(
            func=self.ξ_upd.upd_fn,
            namespace=self,
            input_names=self.ξ_upd.inputs
        )
```

:::{margin} Code  
As the prototype for *noise sources*, all functionality is currently implemented in `GaussianWhiteNoise`.  
:::

```{code-cell} ipython3
---
jupyter:
  source_hidden: true
tags: [hide-input]
---
    @add_to('GaussianWhiteNoise')
    def integrate(self, upto='end'):
        # TODO: Do this in one vectorized operation.
        # REM: All that should be required is to define range_update_function
        self.ξ._compute_up_to(upto)


    ## Copied from sinn.models.Model
    @add_property_to('GaussianWhiteNoise')
    def name(self):
        return getattr(self, '__name__', type(self).__name__)

    @add_property_to('GaussianWhiteNoise')
    def nonnested_histories(self):
        return {nm: getattr(self, nm) for nm in self._hist_identifiers}
    @add_property_to('GaussianWhiteNoise')
    def nonnested_history_set(self):
        return {getattr(self, nm) for nm in self._hist_identifiers}
    @add_property_to('GaussianWhiteNoise')
    def history_set(self):
        return set(chain(self.nonnested_history_set,
                         *(m.history_set for m in self.nested_models_list)))
    @add_property_to('GaussianWhiteNoise')
    def nonnested_kernels(self):
        return {nm: getattr(self, nm) for nm in self._kernel_identifiers}
    @add_property_to('GaussianWhiteNoise')
    def nonnested_kernel_list(self):
        return [getattr(self, nm) for nm in self._kernel_identifiers]
    @add_property_to('GaussianWhiteNoise')
    def kernel_list(self):
        return list(chain(self.nonnested_kernel_list,
                          *(m.kernel_list for m in self.nested_models_list)))
    @add_property_to('GaussianWhiteNoise')
    def nested_models(self):
        return {nm: getattr(self, nm) for nm in self._model_identifiers}
    @add_property_to('GaussianWhiteNoise')
    def nested_models_list(self):
        return [getattr(self, nm) for nm in self._model_identifiers]

    @add_property_to('GaussianWhiteNoise')
    def t0(self):
        return self.time.t0
    @add_property_to('GaussianWhiteNoise')
    def tn(self):
        return self.time.tn
    @add_property_to('GaussianWhiteNoise')
    def t0idx(self):
        return self.time.t0idx
    @add_property_to('GaussianWhiteNoise')
    def tnidx(self):
        return self.time.tnidx
    @add_property_to('GaussianWhiteNoise')
    def tidx_dtype(self):
        return self.time.Index.dtype
    @add_property_to('GaussianWhiteNoise')
    def dt(self):
        return self.time.dt

    @add_to('GaussianWhiteNoise')
    def __getattribute__(self, attr):
        """
        Retrieve parameters if their name does not clash with an attribute.
        """
        # Use __getattribute__ to maintain current stack trace on exceptions
        # https://stackoverflow.com/q/36575068
        if (not attr.startswith('_')      # Hide private attrs and prevent infinite recursion with '__dict__'
            and attr != 'params'          # Prevent infinite recursion
            and attr not in dir(self)     # Prevent shadowing; not everything is in __dict__
            # Checking dir(self.params) instead of self.params.__fields__ allows
            # Parameter classes to use @property to define transformed parameters
            and hasattr(self, 'params') and attr in dir(self.params)):
            # Return either a PyMC3 prior (if in PyMC3 context) or shared var
            # Using sys.get() avoids the need to load pymc3 if it isn't already
            pymc3 = sys.modules.get('pymc3', None)
            param = getattr(self.params, attr)
            if not hasattr(param, 'prior') or pymc3 is None:
                return param
            else:
                try:
                    pymc3.Model.get_context()
                except TypeError:
                    # No PyMC3 model on stack – return plain param
                    return param
                else:
                    # Return PyMC3 prior
                    return param.prior

        else:
            return super(GaussianWhiteNoise,self).__getattribute__(attr)
    
    ## Copied from sinnfull.base.Model
    _compiled_functions: dict=PrivateAttr(default_factory=lambda: {})

    def stationary_stats(
        self,
        params: Union[None, ModelParams, IndexableNamespace, dict]=None,
        _max_cost: int=10
        ) -> Union[ModelParams, IndexableNamespace]:
        """
        Public wrapper for _stationary_stats, which does type casting.
        Returns either symbolic or concrete values, depending on whether `params`
        is symbolic.

        .. Note:: If just one of the values of `params` is symbolic, _all_ of the
           returned statistics will be symbolic.

        Parameters
        -----------
        params: Any value which can be used to construct a `cls.Parameters`
            instance. If `None`, `self.params` is used.
        _max_cost: Default value should generally be fine. See
            `theano_shim.graph.eval` for details.
        """
        if params is None:
            params = self.params
        symbolic_params = shim.is_symbolic(params)
        # Casting into cls.Parameters is important because cls.Parameters
        # might use properties to define transforms of variables
        if not isinstance(params, cls.Parameters):
            params = cls.Parameters(params)
        stats = self._stationary_stats(params)
        if not symbolic_params:
            stats = shim.eval(stats)
        return stats

    def stationary_stats_eval(self):
        """
        Equivalent to `self.stationary_stats(self.params).eval()`, with the
        benefit that the compiled function is cached. This saves the overhead
        of multiple calls to `.eval()` if used multiple times.
        """
        # TODO: Attach the compiled function to the class ?
        #       Pro: -only ever compiled once
        #       Con: -would have to use placeholder vars instead of the model's vars
        #            -how often would we really have more than one model instance ?
        compile_key = 'stationary_stats'
        try:
            compiled_stat_fn = self._compiled_functions[compile_key]
        except KeyError:
            # NB: `shim.graph.compile(ins, outs)` works when `outs` is a dictionary,
            #     but only when it is FLAT, with STRING-ONLY keys
            #     So we merge the keys with a char combination unlikely to conflict
            flatstats_out = {"||".join((k,statnm)): statval
                             for k, statdict in self.stationary_stats(self.params).items()
                             for statnm, statval in statdict.items()}
            assert all(len(k.split("||"))==2 for k in flatstats_out)
                # Assert that there are no conflicts with the merge indicator
            compiled_stat_fn = shim.graph.compile([], flatstats_out)
            self._compiled_functions[compile_key] = compiled_stat_fn
        flatstats = compiled_stat_fn()
        # TODO: It seems like it should be possible to ravel a flattened dict more compactly
        stats = {}
        for key, statval in flatstats.items():
            key = key.split("||")
            try:
                stats[key[0]][key[1]] = statval
            except KeyError:
                stats[key[0]] = {key[1]: statval}
        return stats
```

## Stationary state

There are no dynamics to speak of, so the equations for stationary distribution are the same as for the model.

$$\begin{aligned}
ξ(t) &\sim \mathcal{N}(μ, \exp(\log σ)^2) \\
ξ_k &\sim \mathcal{N}(μ, \exp(\log σ)^2/Δt)
\end{aligned}$$

:::{margin} Code
`GaussianWhiteNoise`: Stationary distribution
:::

```{code-cell} ipython3
:tags: [hide-input]

    @add_to('GaussianWhiteNoise')
    def _stationary_stats(self, params: ModelParams):
        return {'ξ': {'mean': params.μ,
                      'std': shim.exp(params.logσ)/shim.sqrt(self.dt)}}
    
    @add_to('GaussianWhiteNoise')
    def stationary_dist(self, params: ModelParams):
        stats = cls.stationary_stats(params)
        with pm.Model() as statdist:
            ξ_stationary = pm.Normal(
                "ξ",
                mu=stats['ξ']['mean'], sigma=stats['ξ']['std'],
                shape=(params.M,)
            )
        return statdist
```

### Variables

|**Model variable**| Identifier | Type | Description |
|--------|--|--|--
|$\xi$   | `ξ` | dynamic variable | Output white noise process |
|$\mu$   | `μ` | parameter | mean of $\xi$ |
|$\log \sigma$| `σ` | parameter | (log of the) std. dev of $\xi$; noise strength |
|$M$ | `M` | parameter | number of independent components (dim $ξ$) |

+++

### Testing

+++

It can be useful to also define a `get_test_parameters` method on the class, to run tests without needing to specify a prior.

:::{margin} Code
`GaussianWhiteNoise`: Test parameters
:::

```{code-cell} ipython3
:tags: [hide-input]

    @add_to('GaussianWhiteNoise')
    @classmethod
    def get_test_parameters(cls, rng: Union[None,int,RNGenerator]=None):
        """
        :param:rng: Any value accepted by `numpy.random.default_rng`.
        """
        rng = np.random.default_rng(rng)
        M = rng.integers(1,5)
        Θ = cls.Parameters(μ=rng.normal(size=(M,)),
                           logσ=rng.normal(size=(M,)),
                           M=M)
        return Θ
```

```{code-cell} ipython3
:tags: [remove-cell]

GaussianWhiteNoise.update_forward_refs()  # Always safe to do this & sometimes required
```

## Example output

```{code-cell} ipython3
if __name__ == "__main__":
    import holoviews as hv
    hv.extension('bokeh')
    
    time = TimeAxis(min=0, max=1, step=2**-7, unit=sinnfull.ureg.s)
    Θ = GaussianWhiteNoise.get_test_parameters(rng=123)
    noise = GaussianWhiteNoise(time=time, params=Θ)
    #model.integrate(upto='end')
    
```

```{code-cell} ipython3
    noise.integrate(upto='end')
```

```{code-cell} ipython3
    traces = [hv.Curve(trace, kdims=['time'], vdims=[f'u{i}'])
              for i, trace in enumerate(noise.ξ.traces)]
    hv.Layout(traces)
```

```{code-cell} ipython3

```
