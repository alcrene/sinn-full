# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,md:myst
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
# # Gaussian white noise
#
# The Gaussian white noise model is mostly useful as an input to other models, but can also be inferred directly to test algorithms.

# %% [markdown]
# :::{attention}
# `GaussianWhiteNoise` is a **noise source**, which for practical and technical reasons we distinguish from *models*. One can think of noise sources as models with no dynamics – since each time point is independent, they can be generated with one vectorized call to the random number generator. The extent to which this distinction is useful is still being worked out, but it at least seems to simplify implementations.
#
# At present noise sources do **not** inherit from `sinnfull.models.Model`, although much of the API is reproduced. This may change in the future.
# :::

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')

# %% tags=["hide-cell"]
from typing import Any, Optional, Union
import numpy as np
import theano_shim as shim
from mackelab_toolbox.typing import (
    FloatX, Shared, Array, AnyRNG, RNGenerator,
    IndexableNamespace as IndexableNamespaceBase)
# Move with NoiseSource:
from pydantic import BaseModel, PrivateAttr
import sys, inspect
from types import SimpleNamespace
from itertools import chain
from sinn.histories import History
from sinn.models import PendingUpdateFunction

from sinn.models import ModelParams, Model, updatefunction, initializer
from sinn.histories import TimeAxis, Series, HistoryUpdateFunction

from sinnfull.utils import add_to, add_property_to
from sinnfull.models.base import Model, Param
from sinnfull.typing_ import IndexableNamespace

# %%
__all__ = ['GaussianWhiteNoise']


# %% [markdown]
# ## Model equations
#
# This is a model for $M$ independent noise processes formally given by:
#
# :::{math}
# :label: eq:gwn-def
# \begin{aligned}
# ξ^m(t) &\sim \mathcal{N}(μ^m, \cdot) \\
# \langle (ξ^m(t)-μ^m)(ξ^m(t')-μ^m) \rangle &= {(σ^m)}^2 δ(t-t')
# \end{aligned}
# :::
#
# where $m \in \{1, \dotsc, M\}$ and $μ^m, σ^m, ξ^m(t) \in \mathbb{R}$. In this form the standard deviation of $ξ^m(t)$ is ill-defined; this is easier to do with the discretized process (see Eq. [ξ_k](eq:gwn-discretized)).
# In the equations which follow, as well as the code, we drop the index $m$ and treat $μ$, $σ$ and $ξ$ as arrays to which operations are applied element wise.
#
# :::{Note}
# Below, we write the process explicitely as a function of $\log σ$.
# We do this because $σ$ is a scale parameters that may vary over many orders of magnitude, and we've found that inferrence works best for such parameters when performed in log space. Moreover, since it is strictly positive, this transformation is well-defined and bijective.
# :::
#
# ### Discretized form
#
# Formally, we define $ξ_k$ as
#
# $$ξ_k := \frac{1}{Δt} \int_{t_k}^{t_k+Δt} ξ(t') dt' \,.$$
#
# The consistency condition
#
# $$\frac{1}{Δt} \int_{t_k}^{t_k+Δt}\!\! ξ(t') dt' = \frac{1}{2 Δt} \int_{t_k}^{t_k+Δt/2}\!\!\! ξ(t') dt' + \frac{1}{2 Δt} \int_{t_k+Δt/2}^{t_k+Δt} ξ(t') dt'$$
#
# requires
# :::{math}
# :label: eq:gwn-discretized
# \begin{gathered}
# ξ_k \sim \mathcal{N}\left(μ, \exp(\log σ)^2 \middle/ Δt\right) \,,
# \end{gathered}
# :::
#
# which is the definition we use in the implementation.
#
# :::{margin} Code
# `GaussianWhiteNoise`: Parameters
# `GaussianWhiteNoise`: Update equation
# :::

# %% tags=["hide-input"]
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

    def initialize(self, initializer=None):
        return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ξ.update_function = HistoryUpdateFunction(
            func=self.ξ_upd.upd_fn,
            namespace=self,
            input_names=self.ξ_upd.inputs
        )
        self.ξ.range_update_function = self.ξ.update_function


# %% [markdown]
# :::{margin} Code
# As the prototype for *noise sources*, all functionality is currently implemented in `GaussianWhiteNoise`.
# :::

    # %% tags=["hide-input"] jupyter={"source_hidden": true}
    @add_to('GaussianWhiteNoise')
    def integrate(self, upto='end'):
        # TODO: Do this in one vectorized operation.
        # REM: All that should be required is to define range_update_function
        self.ξ._compute_up_to(upto)

    ## HACKS to match the sinn.Model interface
    # These _might_ fix themselves if we inherit for some base Model class

    @property
    def _pending_update_functions(self):
        return [e for e in self.__dict__.values()
                if isinstance(e, PendingUpdateFunction)]

    ## Copied from sinn.models.Model

    @add_to('GaussianWhiteNoise')
    def __str__(self):
        name = self.name
        return "Model '{}' (t0: {}, tn: {}, dt: {})" \
            .format(name, self.t0, self.tn, self.dt)

    @add_to('GaussianWhiteNoise')
    def __repr__(self):
        return f"<{str(self)}>"

    @add_to('GaussianWhiteNoise')
    def _repr_html_(self):
        summary = self.get_summary()
        topdivline = '<div style="margin-bottom: 0; border-left: solid black 1px; padding-left: 10px">'
        modelline = f'<p style="margin-bottom:0">Model <b>{summary.model_name}</b>'
        if isinstance(self, Model):  # Eventually we want to be able to be able to call this on the class
            modelline += f' (<code>t0: {self.t0}</code>, <code>tn: {self.tn}</code>, <code>dt: {self.dt}</code>)'
        modelline += '</p>'
        topulline = '<ul style="margin-top: 0;">'
        stateline = '<li>State variables: '
        stateline += ', '.join([f'<code>{v}</code>' for v in summary.state_vars])
        stateline += '</li>'
        paramblock = '<li>Parameters\n<ul>\n'
        if isinstance(summary.params, IndexableNamespaceBase):
            for name, val in summary.params:
                paramblock += f'<li><code> {name}={val}</code></li>\n'
        else:
            for name in summary.params:
                paramblock += f'<li><code> {name}</code></li>\n'
        paramblock += '</ul>\n</li>'
        updfnblock = ""
        for varname, upd_code in summary.update_functions.items():
            updfnblock += f'<li>Update function for <code>{varname}</code>\n'
            updfnblock += f'<pre>{upd_code}</pre>\n</li>\n'
        if not summary.nested_models:
            nestedblock = ""
        else:
            nestedblock = '<li>Nested models:\n<ul>'
            for nestedmodel in summary.nested_models:
                nestedblock += f'<li>{nestedmodel._repr_html_()}</li>'
            nestedblock += '</ul></li>\n'

        return "\n".join([
            topdivline, modelline, topulline, stateline,
            paramblock, updfnblock, nestedblock,
            "</ul>", "</div"
        ])


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

    @add_property_to('GaussianWhiteNoise')
    def name(self):
        return getattr(self, '__name__', type(self).__name__)

    @add_property_to('GaussianWhiteNoise')
    def nonnested_histories(self):
        return {nm: getattr(self, nm) for nm in self._hist_identifiers}
    @add_property_to('GaussianWhiteNoise')
    def nonnested_history_set(self):
        return {getattr(self, nm) for nm in self._hist_identifiers}
    @property
    def nested_histories(self):
        return {**self.nonnested_histories,
                **{f"{submodel_nm}.{hist_nm}": hist
                   for submodel_nm, submodel in self.nested_models.items()
                   for hist_nm, hist in submodel.nested_histories.items()}}
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

    # >>>>>
    @add_property_to('GaussianWhiteNoise')
    def statehists(self):
        return []
    # <<<<<

    @add_property_to('GaussianWhiteNoise')
    def unlocked_statehists(self):
        return (h for h in self.statehists if not h.locked)

    @add_property_to('GaussianWhiteNoise')
    def locked_statehists(self):
        return (h for h in self.statehists if h.locked)

    @add_property_to('GaussianWhiteNoise')
    def unlocked_histories(self):
        return (h for h in self.history_set if not h.locked)

    @add_property_to('GaussianWhiteNoise')
    def unlocked_nonstatehists(self):
        return (h for h in self.nonstatehists if not h.locked)

    @add_property_to('GaussianWhiteNoise')
    def rng_inputs(self):
        rng_inputs = []
        for h in self.unlocked_histories:
            for nm in h.update_function.input_names:
                inp = getattr(h.update_function.namespace, nm)
                if (isinstance(inp, shim.config.RNGTypes)
                    and inp not in rng_inputs):
                    rng_inputs.append(inp)
        return rng_inputs

    @add_property_to('GaussianWhiteNoise')
    def rng_hists(self):
        rng_hists = []
        for h in self.unlocked_histories:
            for nm in h.update_function.input_names:
                inp = getattr(h.update_function.namespace, nm)
                if isinstance(inp, shim.config.RNGTypes):
                    rng_hists.append(h)
                    break
        return rng_hists

    @add_to('GaussianWhiteNoise')
    def get_min_tidx(self, histories: Sequence[History]):
        try:
            return min(h.cur_tidx.convert(self.time.Index)
                       for h in histories)
        except IndexError as e:
            raise IndexError(
                "Unable to determine a current index for "
                f"{self.name}. This usually happens accessing "
                "`cur_tidx` before a model is initialized.") from e

    @add_property_to('GaussianWhiteNoise')
    def cur_tidx(self):
        return self.get_min_tidx(self.statehists)

    #def curtidx_var(self):

    #def stopidx_var(self):

    #def batchsize_var(self):

    @add_to('GaussianWhiteNoise')
    def get_num_tidx(self, histories: Sequence[History]):
        if not self.histories_are_synchronized():
            raise RuntimeError(
                f"Histories for the {self.name}) are not all "
                "computed up to the same point. The compilation of the "
                "model's integration function is ill-defined in this case.")
        tidx = self.get_min_tidx(histories)
        if not hasattr(self, '_num_tidx'):
            object.__setattr__(
                self, '_num_tidx',
                shim.shared(np.array(tidx, dtype=self.tidx_dtype),
                            f"t_idx ({self.name})") )
        else:
            self._num_tidx.set_value(tidx)
        return self._num_tidx

    @add_property_to('GaussianWhiteNoise')
    def num_tidx(self):
        return self.get_num_tidx(self.statehists)

    @add_to('GaussianWhiteNoise')
    def histories_are_synchronized(self):
        try:
            tidcs = [h.cur_tidx.convert(self.time.Index)
                     for h in self.unlocked_statehists]
        except IndexError:
            # If conversion fails, its because the converted index would be out
            # of the range of the new hist => hists clearly not synchronized
            return False
        locked_tidcs = [h.cur_tidx.convert(self.time.Index)
                        for h in self.locked_statehists]
        if len(tidcs) == 0:
            return True
        earliest = min(tidcs)
        latest = max(tidcs)
        if earliest != latest:
            return False
        elif any(ti < earliest for ti in locked_tidcs):
            return False
        else:
            return True

    @add_to('GaussianWhiteNoise')
    def get_summary(self, hists=None):
        summary = SimpleNamespace()
        summary.model_name = self.name
        State = getattr(self, 'State', None)
        if State:
            summary.state_vars = list(State.__annotations__)
        else:
            summary.state_vars = []
        if isinstance(self, type):
            Parameters = getattr(self, 'Parameters', None)
            if Parameters:
                summary.params = list(Parameters.__annotations__)
            else:
                summary.params = []
        else:
            params = getattr(self, 'params', None)
            if params:
                summary.params = params.get_values()
            else:
                summary.params = {}
        summary.update_functions = self.get_update_summaries(hists)
        summary.nested_models = list(self.nested_models.values())
        return summary

    @add_to('GaussianWhiteNoise')
    def get_update_summaries(self, hists=None) -> Dict[str,str]:
        """
        For selected histories, return a string summarizing the update function.
        By default, the histories summarized are those of `self.state`.
        May be called on the class itself, or an instance.

        Parameters
        ----------
        hists: list | tuple of histories or str
            List of histories to summarize.
            For each history given, retrieves its update function.
            Alternatively, a history's name can be given as a string

        Returns
        -------
        Dict[str,str]: {hist name: hist upd code}
        """
        # Default for when `hists=None`
        if hists is None:
            # >>>> Difference wrt `Model`: No State
            State = getattr(self, 'State', None)
            if State:
                hists = list(self.State.__annotations__.keys())
            else:
                hists = []
            # <<<<
        # Normalize to all strings; take care that identifiers may differ from the history's name
        histsdict = {}
        nonnested_hists = getattr(self, 'nonnested_histories', {})
        for h in hists:
            if isinstance(h, History):
                h_id = None
                for nm, hist in nonnested_hists.items():
                    if hist is h:
                        h_id = nm
                        break
                if h_id is None:
                    continue
                h_nm = h.name
            else:
                assert isinstance(h, str)
                if h not in self.__annotations__:
                    continue
                h_id = h
                h_nm = h
            histsdict[h_id] = h_nm
        hists = histsdict
        funcs = {pending.hist_nm: pending.upd_fn
                 for pending in self._pending_update_functions}
        if not all(isinstance(h, str) for h in hists):
            raise ValueError(
                "`hists` must be a list of histories or history names.")

        # For each function, retrieve its source
        srcs = {}
        for hist_id, hist_nm in hists.items():
            fn = funcs.get(hist_id, None)
            if fn is None:
                continue
            src = inspect.getsource(fn)
            # Check that the source defines a function as expected:
            # first non-decorator line should start with 'def'
            for line in src.splitlines():
                if line.strip()[0] == '@':
                    continue
                elif line.strip()[:3] != 'def':
                    raise RuntimeError(
                        "Something went wrong when retrieve an update function's source. "
                        "Make sure the source file is saved and try reloading the Jupyter "
                        "notebook. Source should start with `def`, but we got:\n" + src)
                else:
                    break
            # TODO: Remove indentation common to all lines
            if hist_id != hist_nm:
                hist_desc = f"{hist_id} ({hist_nm})"
            else:
                hist_desc = hist_id
            srcs[hist_desc] = src.split('\n', 1)[1]
                # 'split' is used to replace the `def` line: callers use
                # the `hist_desc` value to create a more explicit string
        return srcs

    @add_to('GaussianWhiteNoise')
    def summarize(self, hists=None):
        nested_models = self._model_identifiers
        if isinstance(self, type):
            nameline = "Model '{}'".format(self.name)
            paramline = "Parameters: " + ', '.join(self.Parameters.__annotations__) + "\n"
            if len(nested_models) == 0:
                nestedline = ""
            else:
                nestedline = "Nested models:\n    " + '\n    '.join(nested_models)
                nestedline = nestedline + "\n"
            nested_summaries = []
        else:
            # >>>>
            #assert isinstance(self, Model)
            assert isinstance(self, BaseModel)  # -> NoiseSource ?
            # <<<<
            nameline = str(self)
            if hasattr(self, 'params'):
                paramline = f"Parameters: {self.params}\n"
            else:
                paramline = "Parameters: None\n"
            if len(nested_models) == 0:
                nestedline = ""
            else:
                nestedlines = [f"    {attr} -> {type(cls).__name__}"
                               for attr, cls in self.nested_models.items()]
                nestedline = "Nested models:\n" + '\n'.join(nestedlines) + "\n"
            nested_summaries = [model.summarize(hists)
                                for model in self.nested_models.values()]
        nameline += '\n' + '-'*len(nameline) + '\n'  # Add separating line under name
        # >>>> Difference wrt `Model`: No State
        State = getattr(self, 'State', None)
        if State:
            stateline = "State variables: " + ', '.join(State.__annotations__)
            stateline = stateline + "\n"
        else:
            stateline = ""
        # <<<<
        summary = (nameline + stateline + paramline + nestedline
                   + '\n' + self.summarize_updates(hists))
        return '\n'.join((summary, *nested_summaries))

    @add_to('GaussianWhiteNoise')
    def get_tidx(self, t, allow_rounding=False):
        # Copied from History.get_tidx
        if self.time.is_compatible_value(t):
            return self.time.index(t, allow_rounding=allow_rounding)
        else:
            # assert self.time.is_compatible_index(t)
            assert shim.istype(t, 'int')
            if (isinstance(t, sinn.axis.AbstractAxisIndexDelta)
                and not isinstance(t, sinn.axis.AbstractAxisIndex)):
                raise TypeError(
                    "Attempted to get the absolute time index corresponding to "
                    f"{t}, but it is an index delta.")
            return self.time.Index(t)

    def index_interval(self, Δt, allow_rounding=False):
        return self.time.index_interval(value, value2,
                                        allow_rounding=allow_rounding,
                                        cast=cast)

    def get_time(self, t):
        # Copied from History
        # NOTE: Copy changes to this function in Model.get_time()
        # TODO: Is it OK to enforce single precision ?
        if shim.istype(t, 'int'):
            # Either we have a bare int, or an AxisIndex
            if isinstance(t, sinn.axis.AbstractAxisIndex):
                t = t.convert(self.time)
            elif isinstance(t, sinn.axis.AbstractAxisIndexDelta):
                raise TypeError(f"Can't retrieve the time corresponding to {t}: "
                                "it's a relative, not absolute, time index.")
            return self.time[t]
        else:
            assert self.time.is_compatible_value(t)
            # `t` is already a time value -> just return it
            return t

    @add_to('GaussianWhiteNoise')
    def lock(self):
        for hist in self.history_set:
            hist.lock()

    @add_to('GaussianWhiteNoise')
    def clear(self,after=None):
        shim.reset_updates()
        if after is not None:
            after = self.time.Index(after)
            for hist in self.unlocked_histories:
                hist.clear(after=after.convert(hist.time.Index))
        else:
            for hist in self.unlocked_histories:
                hist.clear()

    def eval(self, max_cost :Optional[int]=None, if_too_costly :str='raise'):
        for h in self.history_set:
            h.eval(max_cost, if_too_costly)


    ## Copied from sinnfull.base.Model
    _compiled_functions: dict=PrivateAttr(default_factory=lambda: {})

    @add_to('GaussianWhiteNoise')
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
        params: Any value which can be used to construct a `self.Parameters`
            instance. If `None`, `self.params` is used.
        _max_cost: Default value should generally be fine. See
            `theano_shim.graph.eval` for details.
        """
        if params is None:
            params = self.params
        symbolic_params = shim.is_symbolic(params)
        # Casting into self.Parameters is important because self.Parameters
        # might use properties to define transforms of variables
        if not isinstance(params, self.Parameters):
            params = self.Parameters(params)
        stats = self._stationary_stats(params)
        if not symbolic_params:
            stats = shim.eval(stats)
        return stats

    @add_to('GaussianWhiteNoise')
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

# %% [markdown]
# ## Stationary state
#
# This model has no dynamics, so the equations for stationary distribution are the same as for the model.
#
# $$\begin{aligned}
# ξ_k &\sim \mathcal{N}(μ, \exp(\log σ)^2/Δt)
# \end{aligned}$$
#
# :::{margin} Code
# `GaussianWhiteNoise`: Stationary distribution
# :::

    # %% tags=["hide-input"]
    @add_to('GaussianWhiteNoise')
    def _stationary_stats(self, params: ModelParams):
        dt = self.dt
        dt = getattr(dt, 'magnitude', dt)  # In case 'dt' is a Pint or Quantities
        return {'ξ': {'mean': params.μ,
                      'std': shim.exp(params.logσ)/shim.sqrt(dt)}}

    @add_to('GaussianWhiteNoise')
    def stationary_dist(self, params: ModelParams):
        stats = self.stationary_stats(params)
        with pm.Model() as statdist:
            ξ_stationary = pm.Normal(
                "ξ",
                mu=stats['ξ']['mean'], sigma=stats['ξ']['std'],
                shape=(params.M,)
            )
        return statdist

# %% [markdown]
# ## Variables
#
# |**Model variable**| Identifier | Type | Description |
# |--------|--|--|--
# |$\xi$   | `ξ` | dynamic variable | Output white noise process |
# |$\mu$   | `μ` | parameter | mean of $\xi$ |
# |$\log \sigma$| `σ` | parameter | (log of the) std. dev of $\xi$; noise strength |
# |$M$ | `M` | parameter | number of independent components (dim $ξ$) |

# %% [markdown]
# ### Testing

# %% [markdown]
# It can be useful to also define a `get_test_parameters` method on the class, to run tests without needing to specify a prior.
#
# :::{margin} Code
# `GaussianWhiteNoise`: Test parameters
# :::

    # %% tags=["hide-input"]
    @add_to('GaussianWhiteNoise')
    @classmethod
    def get_test_parameters(cls, rng: Union[None,int,RNGenerator]=None):
        """
        :param:rng: Any value accepted by `numpy.random.default_rng`.
        """
        rng = np.random.default_rng(rng)
        #M = rng.integers(1,5)
        M = 2  # Currently no way to ensure submodels draw the same M
        Θ = cls.Parameters(μ=rng.normal(size=(M,)),
                           logσ=rng.normal(size=(M,)),
                           M=M)
        return Θ

# %% tags=["remove-cell"]
Model.register(GaussianWhiteNoise)
from mackelab_toolbox import iotools
iotools.register_datatype(GaussianWhiteNoise)  # TODO: Do in NoiseSource; c.f. Models.__init_subclass__
GaussianWhiteNoise.update_forward_refs()  # Always safe to do this & sometimes required

# %% [markdown]
# ## Example output

# %%
if __name__ == "__main__":
    from tqdm.auto import tqdm
    from scipy import stats
    import holoviews as hv
    hv.extension('bokeh')

    time = TimeAxis(min=0, max=10, step=2**-6, unit=sinnfull.ureg.s)
    Θ = GaussianWhiteNoise.get_test_parameters(rng=123)
    noise = GaussianWhiteNoise(time=time, params=Θ)
    #model.integrate(upto='end')


    # %%
    noise.integrate(upto='end')

    # %%
    traces = [hv.Curve(trace, kdims=['time'], vdims=[f'ξ{i}'])
              for i, trace in enumerate(noise.ξ.traces)]
    hv.Layout(traces)

# %% [markdown]
# ## Testing

# %% [markdown]
# The integral of this process is a Brownian motion:
#
# $$\int_0^T ξ(t) dt \sim \mathcal{N}(μT, σ^2 T)$$
#
# We can use this to test the correctness of our implementation: simulate the `GaussianWhiteNoise` multiple times to form a Monte Carlo estimate of $p\left(\int_0^T ξ(t) dt\right)$, and compare with the expression above.

    # %%
    int_ξ = []
    N_sims = 100
    T = (noise.tn-noise.t0)  # time we integrate for
    for i in tqdm(range(N_sims)):
        noise.clear()
        noise.integrate('end')
        int_ξ.append( noise.ξ.data.sum(axis=0)*noise.dt )

    # %%
    int_ξ_arr = np.array(int_ξ)
    panels = []
    for i in range(noise.M):
        counts, edges = np.histogram(int_ξ_arr[:,i],
                                     bins='auto',
                                     density=True)
        histogram = hv.Histogram((counts, edges),
                                 kdims=[f"∫ξ{i}"], vdims=[f"p(∫ξ{i})"],
                                 label="Monte Carlo"
                                )
        ξarr = np.linspace(edges[0], edges[-1], 100)
        pdf_vals = stats.norm(loc=noise.μ*T.magnitude,
                              scale=np.exp(noise.logσ)*np.sqrt(T.magnitude)
                             ).pdf(ξarr)
        pdf = hv.Curve(zip(ξarr, pdf_vals),
                       kdims=[f"∫ξ{i}"], vdims=[f"p(∫ξ{i})"],
                       label="Theory")
        panels.append(histogram * pdf.opts(color='orange'))
    hv.Layout(panels)

# %%
