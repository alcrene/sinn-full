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
# # Alternated SGD optimizer
# $
# \newcommand{\tran}{\mathrm{\intercal}}
# \newcommand{\step}[2]{#2^{(#1)}}
# \newcommand{\mathop}[1]{\mathrm{#1}}
# $
# This module defines an _alternated SGD_ algorithm for optimizing the likelihood of a model with both unknown parameters and latents. For usage instructions, see the accompanying [example](./alternated_sgd_tests_examples).
#
# The algorithm attempts to be as straightforward and intuitive as possible, in the sense that it closely approximates simply taking the gradient of the joint likelihood over latents and parameters. The choice of alternating between parameters and latents is to reduce the cost of computing gradients on the parameters.

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('theano')
import logging
logger = logging.getLogger(__name__)

# %% [markdown] tags=["remove-cell"]
# The `diagnostic_hooks` flag indicates whether to save extra attributes for inspecting the optimization steps. By default, it is only true when running within this notebook, but can be overwritten by assigning ``True`` to the module attribute.

# %%
from sinnfull import config
import sinnfull.diagnostics
sinnfull.diagnostics.set(__name__ == "__main__")

# %% tags=["hide-input"]
from warnings import warn
from types import SimpleNamespace
from typing import Union, Optional, Any, Callable, List, Tuple, Dict
from collections import namedtuple

import numpy as np
import xarray as xr
import pymc3 as pm

from pydantic import BaseModel, validator, root_validator, constr, PrivateAttr

import theano_shim as shim

import mackelab_toolbox as mtb
from mackelab_toolbox.typing import RNGenerator, Integral, FloatX
import mackelab_toolbox.optimizers
import mackelab_toolbox.serialize

# %% tags=["hide-input"]
import sinn
from sinn.histories import History
from sinn.models import initializer, ModelParams
from sinn.utils import unlocked_hists
from sinn.diskcache import DiskCache

import sinnfull.utils as utils
from sinnfull.sampling import sample_batch_starts, SegmentSampler
from sinnfull.parameters import ParameterSet
from sinnfull.utils import add_to, add_property_to
from sinnfull.models.base import Prior, AccumulatedObjectiveFunction
from sinnfull.optim.base import Optimizer, OptimParams, OptimizerStatus, clip_gradients
from sinnfull.optim.convergence_tests import ConvergenceTest
from sinnfull.optim.recorders import Recorder, DiagnosticRecorder
    # Will be deprecated: we no longer needto attach recorders

# %% tags=["remove-cell"]
if __name__ == "__main__":
    from IPython.display import display


# %% [markdown]
# :::{admonition} Inherited notation
# :class: dropdown
#
# For reference, we copy below notation defined by the [base class](../base).
#
# **Notation conventions**
#
# (Repeated from the [base class](../base))
# - Uppercase $\Theta$ / `Θ`: set of parameters
# - Lowercase $\theta$ / `θ`: parameter, i.e. $\theta \in \Theta$
# - Lowercase $\eta$ / `η`: latent variable(s)
# - $K$: Length in time bins
# - $T$: Length in continuous time
#
# **Important optimizer attributes**
#
# These can be retrieved as `optimizer.<attr name>`.
#
# `Θ`
# ~ OptimParams; set of shared variables in _optimization space_.
# ~ Use the `init_params` argument to set it
#
# `fit_hyperparams`
# ~ nested dict
#
# `observed_hists`
# ~ Set of `History` instances for which we have data.
#
# `latent_hists`
# ~ Set of `History` instances we need to infer.
#
# `stepi`
# ~ Number of optimization steps already performed
#
# :::

# %% [markdown]
# ### Forward and backward gradients
#
# :::{note}
# For historical reasons, the algorithm is presented using batches for both the gradient w.r.t. parameters and the gradient w.r.t. latents. In practice, we believe that using batches for the latents is unlikely to be helpful, and recommend setting $\mathcal{B}_i$ equal to the entire data segment in the expression for $g_η$ below.
# :::
#
# We compute the gradient of the log likelihood using back-propagation through time (BPTT). As usual, we don't do this over the entire data set, but on batches of sequential data. A batch $\mathcal{B}_i$ runs from $k=b_i$ to $k=b_i + K_b + K_r$, where $K_b$ is the length of data used to compute the likelihood, and $K_r$ a time bin offset to allow for BPTT; this offset should be comparable with the **r**elaxation time scale of the dynamics. It is added either to the beginning or the end of the batch, depending on whether we optimize parameters or latents (see below). In order to be bin-size agnostic, in the algorithm $K_b$ and $K_r$ are specified in continuous time as $T_b$ and $T_r$, with $K_b := \lfloor T_b/Δt \rfloor$ and $K_r := \lfloor T_r/Δt \rfloor$.
#
#
# ```{math}
# :label: eq:gradient-params
#   \step{s}{g}_θ := \widehat{\nabla_θ l(θ;η)}\hphantom{{}_{[\mathcal{B}_i]}} = \sum_{\mspace{-30mu}\rlap{k=b^θ_s+K^θ_r}}^{\mspace{-30mu}\rlap{b^θ_s+K^θ_r+K^θ_b}} \nabla_θ l_k(θ) \,,
# ```
# ```{math}
# :label: eq:gradient-latents
#   \step{s}{g}_{η[\mathcal{B}_i]} := \widehat{\nabla_{η[\mathcal{B}_i]} l(θ;η)} = \nabla_{b^η_s:b^η_s+K^η_b} \;l_{b^η_s:b^η_s+K^η_b+K^η_r}(θ) \,.
# ```

# %% [markdown] tags=["remove-cell"]
# \begin{align}
#   \step{s}{g}_θ := \widehat{\nabla_θ l(θ;η)}\hphantom{{}_{[\mathcal{B}_i]}} &= \sum_{\mspace{-30mu}\rlap{k=b^θ_s+K^θ_r}}^{\mspace{-30mu}\rlap{b^θ_s+K^θ_r+K^θ_b}} \nabla_θ l_k(θ) \,, \\
#   \step{s}{g}_{η[\mathcal{B}_i]} := \widehat{\nabla_{η[\mathcal{B}_i]} l(θ;η)} &= \nabla_{b^η_s:b^η_s+K^η_b} \;l_{b^η_s:b^η_s+K^η_b+K^η_r}(θ) \,.
# \end{align}

# %% [markdown]
# Here $η_{[\mathcal{B}_i]}$ is shorthand for the set of latent variables contained within the batch $\mathcal{B}_i$. The expression for $\step{s}{g}_{η[\mathcal{B}_i]}$ can be understood as follows: evaluate the log likelihood for the time interval $[b^η_s:b^η_s+K^η_b+K^η_r]$ and take its gradient with respect to the latent variables at times $[b^η_s:b^η_s+K^η_b]$ (recall that latent variables are dynamic, so effectively we have a set of latent variables for each time point). The extra length $K^η_r$ in the likelihood is a heuristic for the time required for the likelihood to accumulate enough information on the latents. The expression for $\step{s}{g}_θ$ can be given in a similar form as $\step{s}{g}_η$, but we find the form given above to be more transparent.
#
# We note that in contrast to usual RNN training, we _don't_ reinitialize the latent variables for each batch, since the latents are also being optimized.
#
# The reason we compute separate gradients for latent variables and parameters lies in the causal nature of the dynamical equations: _past_ latent states determine _future_ observations. This means that the initial segment of a batch provides little information on the observations (and therefore the parameters), while the final segment provides little information on the value of the latent variables within that segment. Moreover, there are a lot more latent variables than parameters to infer, and compared to parameters, the gradient with respect to the latents is relatively cheap to evaluate. So it makes sense to evaluate $g_η$ on more or larger batches.
#
# For the reasons stated above, $g_θ$ is evaluated on a forward pass (simultaneously computing the model) and therefore called the *forward gradient*; $g_η$ is evaluated on a backward pass (on an already computed model) and referred to as a *backward gradient*.

# %% [markdown]
# ## Definition: `AlternatedSGD`
#
# **TODO**: Remove support for attaching recorders. This avoids requiring one to detach recorders before saving, and is better done in a Task anyway. Files which still depend on this:
#
# - diagnose/Check uniformity of gradients
# - diagnose/Inspect gradients
# - diagnose/Inspect hyperparams

# %%
class AlternatedSGD(Optimizer):
    __slots__ = ('_compile_context', '_k_vars',
                 'logp', 'update_θ', 'update_η',
                 '_latent_cache', '_latent_cache_path', '_current_segment_key',
                 'compiled',
                 'diagnostics_recorders'  # Don't include in Pydantic model
                )

    logp_params        : Callable[[Integral, Integral], Tuple[List[FloatX],dict]]
    logp_latents       : Callable[[Integral, Integral], Tuple[List[FloatX],dict]]
    prior_params       : Prior
    prior_latents      : Optional[Prior]=None
    param_optimizer    : Callable=None
    latent_optimizer   : Callable=None

    # DEPRECATION: Everything with 'nodyn' should probably be deprecated
    logp_default_nodyn : Optional[Callable[[Integral], FloatX]]=None
    logp_latents_nodyn : Optional[Callable[[Integral, Integral], Tuple[List[FloatX],dict]]]=None
        # NB: logp_latents_nodyn is not Optional if we use batched latent updates
    _recorders          : Dict[str,Recorder]=PrivateAttr(default_factory=lambda:{})
        # _recorders attribute will be deprecated
    # TODO: Also move convergence tests to the task
    convergence_tests  : List[ConvergenceTest]=[]

    # # Entries to be excluded from dict, JSON, but included when copying
    # update_θ: Optional[Callable]=None
    # update_η: Optional[Callable]=None
    # _latent_cache: DiskCache
    # _k_vars: Any=SimpleNamespace()
    # _current_segment_key: Optional[tuple]

    class Config:
        arbitrary_types_allowed=True  # TODO: Remove
        allow_population_by_field_name=True  # When Θ gets exported, accept it as parameter
        fields={'Θ': {'alias': 'init_params'},
                'logp_default_nodyn': {'alias': 'logp_nodyn'}}
        json_encoders = sinnfull.json_encoders

    """
    Important attributes
    --------------------
    Θ     : OptimParams; set of shared variables in _optimization space_
    fit_hyperparams: nested dict
    observed_hists
    latent_hists
    stepi: Number of optimization steps performed

    .. remark:: For convenience in test code, the `data_segments` accepts
       arbitrary iterables. However these are not serializable, so when possible
       use `SegmentSampler` objects.

    Parameters
    ----------
    model:
    rng:
    data_segments:
    observed_hists:
    latent_hists:
    fit_hyperparams:
    update_hyperparams:
        See `Optimizer`
    logp_params:
    logp_latents:
    logp_latents_nodyn:
        Accumulator functions; (startidx, batch_size) -> (accumulated_var, updates)
        `logp_params` is used to compile the parameter updates;
        `logp_latents` is used to compile the latent updates
        `logp_latents_nodyn` (deprecated) is used for the rightmost latent update;
          it should exclude contributions from the latent dynamics to the cost.
    logp: Function (tidx) -> Real
        Serves as a default for either `logp_params` or `logp_latents`.
        Note that this is not an accumulator; it will be wrapped with
        either `model.accumulate` (`logp_params`) or
        `model.static_accumulate` (`logp_latents`).
        If `logp` is not provided, and the model defines a `logp`
        attribute, then that is used as default.
    logp_nodyn: Function (tidx) -> Real
        Serves as default for `logp_latents_nodyn`.
        Not an accumulator; it will be wrapped with `model.static_accumulate`.
        If `logp_nodyn` is not provided, and the model defines a `logp_nodyn`
        attribute, then that is used as default.
    param_optimizer: Function (cost, params, λθ, **kwargs) -> update dict
        Signature must match that of `self.default_param_optimizer`.
        If unspecified, `self.default_param_optimizer` is used; this is an Adam optimizer.
    prior_params: Prior
        Instances of `Prior` are converted to `Regularizer`.
        Arbitrary functions are also allowed, but they must have the indicated
        signature (this is not checked.)
    prior_latents:
        Not implemented at this time.
    latent_optimizer: Function (cost, λη, **kwargs) -> dict of update dicts
        Signature must match that of `self.default_latent_optimizer`.
        If unspecified, `self.default_latent_optimizer` is used; this is a plain SGD optimizer.

    stepi: int (Internal variable)
        The number of completed passes.
    orig_fit_hyperparams: ParameterSet (Internal variable)
        A copy of `fit_hyperparams` storing the original values.
        During optimization, functions may change the hyperparameters, for example
        to implement a learning rate schedule. This copy retains the original values;
        it is created automatically if not provided.
        (The only case where it *should* be provided is when restarting a previous
        optimization.)
    compiled: bool (Internal variable)
        Flag indicating whether `compile_optimization_functions` has been called.
        Used when deserializing to determine whether to call it automatically.
    """



    ## Deserialize functions passed as string
    @validator('update_hyperparams', 'logp_default', 'logp_default_nodyn',
               'logp_params', 'logp_latents', 'logp_latents_nodyn',
               pre=True)
    def parse_callback_from_str(cls, func):
        if isinstance(func, str):
            func = mtb.serialize.deserialize_function(func)
        return func

    @validator('logp_default', 'logp_default_nodyn',
               'logp_params', 'logp_latents', 'logp_latents_nodyn', pre=True)
    def force_validate_objectives(cls, v):
        """
        I can't figure why, but serialized AccumulatedObjectiveFunction
        (subclass of BaseModel) don't deserialize. This forces them to.
        """
        if isinstance(v, dict):
            v = AccumulatedObjectiveFunction.validate(v)
        return v

    ## Apply precedence rules to logp functions ##
    # Remark: We also allow logp_params & logp_latents to be plain pointwise functions and wrap them accordingly
    # TODO?: Combine the two logp initializers ?
    @initializer('logp_params', always=True)
    def set_logp_params(cls, logp_params, logp_default, model):
        if logp_params is None:
            if logp_default is None:
                logp_default = getattr(model, 'logp', None)
            if logp_default is None:
                raise ValueError("Either `logp_params` or `logp` must be specified.")
            logp_params = logp_default
        acc_params = getattr(logp_params, '_accumulator', None)
        if acc_params is None:
            logp_params = model.accumulate(logp_params)
        elif acc_params != 'accumulate':
            raise ValueError("The accumulator `logp_params` should be wrapped with "
                             f"the 'accumulate' decorator, not '{acc_params}'.")
        return logp_params

    @initializer('logp_latents', always=True)
    def set_logp_latents(cls, logp_latents, logp_default, model):
        if logp_latents is None:
            if logp_default is None:
                logp_default = getattr(model, 'logp', None)
            if logp_default is None:
                raise ValueError("Either `logp_latents` or `logp` must be specified.")
            logp_latents = logp_default
        acc_latents = getattr(logp_latents, '_accumulator', None)
        if acc_latents is None:
            logp_latents = model.static_accumulate(logp_latents)
        elif acc_latents != 'static_accumulate':
            raise ValueError("The accumulator `logp_latents` should be wrapped with "
                             f"the 'static_accumulate' decorator, not '{acc_latents}'.")
        return logp_latents

    @initializer('logp_latents_nodyn', always=True)
    def set_logp_latents_nodyn(cls, logp_latents_nodyn, logp_default_nodyn, model):
        if logp_latents_nodyn is None:
            if logp_default_nodyn is None:
                return None
            #     logp_default_nodyn = getattr(model, 'logp_nodyn', None)
            # if logp_default_nodyn is None:
            #     raise ValueError("Either `logp_latents_nodyn` or `logp_nodyn` must be specified.")
            logp_latents_nodyn = logp_default_nodyn
        acc_latents_nodyn = getattr(logp_latents_nodyn, '_accumulator', None)
        if acc_latents_nodyn is None:
            logp_latents_nodyn = model.static_accumulate(logp_latents_nodyn)
        elif acc_latents_nodyn != 'static_accumulate':
            raise ValueError("The accumulator `logp_latents` should be wrapped with "
                             f"the 'static_accumulate' decorator, not '{acc_latents_nodyn}'.")
        return logp_latents_nodyn

    @validator('prior_latents')
    def prior_latents_not_implemented(cls, v):
        if v is not None:
            raise NotImplementedError("Regularization of latents is not implemented.")
        return v

    @root_validator(skip_on_failure=True)
    def remove_logp_default(cls, values):
        """
        Once logp_params and logp_latents are set, there is no need for
        logp_default. Keeping it just adds noise to the signature.
        """
        values['logp_default'] = None
        return values

    def __init__(self, *args, latent_cache_path=".cache/latents", **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: Use PrivateAttr instead of __setattr__
        object.__setattr__(self, '_latent_cache_path', latent_cache_path)
        object.__setattr__(self, 'compiled', False)
        object.__setattr__(self, 'diagnostic_recorders', {})

    @property
    def latent_cache(self):
        # By only creating the latent_cache when it is needed, we avoid creating a bunch when
        # we are just loading pre-run optimizers to view their result.
        if not hasattr(self, '_latent_cache'):
            object.__setattr__(self, '_latent_cache', DiskCache(self._latent_cache_path))
        return self._latent_cache

    @property
    def batched_latents(self):
        """
        Specify whether to use batches for the latent updates, or to differentiate
        and update the entire trace each time.
        Rule: batched updates are used if the batch size and batch relaxation
        sizes sum to less than the data size.
        """
        Tηb = self.fit_hyperparams['Tηb'] * self.model.time.unit
        Tηr = self.fit_hyperparams['Tηr'] * self.model.time.unit
        return Tηb + Tηr < self.data_segments.T

    def dict(self, *args, **kwargs):
        """
        Do the same thing with logp functions as Optimizer does with histories:
        replace them by their non-accumulator versions, and let the validators
        re-parse them when deserializing.
        """
        Store = namedtuple('Store', ['logp_params', 'logp_latents',
                                     'logp_latents_nodyn'])
        store = Store(self.logp_params, self.logp_latents,
                      self.logp_latents_nodyn)
        assert hasattr(self.logp_params, '_accumulator')
        assert hasattr(self.logp_latents, '_accumulator')
        self.logp_params = self.logp_params.__func__
        self.logp_latents = self.logp_latents.__func__
        if self.logp_latents_nodyn is not None:
            self.logp_latents_nodyn = self.logp_latents_nodyn.__func__

        d = super().dict(*args, **kwargs)

        d.pop('logp', None)

        self.logp_params = store.logp_params
        self.logp_latents = store.logp_latents
        self.logp_latents_nodyn = store.logp_latents_nodyn
        return d

    # TODO: The three functions below will be deprecated
    def add_recorder(self, recorder):
        if not sinnfull.config.diagnostic_hooks:
            logger.warning("Planned deprecation: Attaching recorders directly "
                           "to an optimizer is not necessary: simply call the "
                           "recorder within the same loop which calls `step()`.")
        recorders_dict = (self.diagnostic_recorders if isinstance(recorder, DiagnosticRecorder)
                          else self._recorders)
        if recorder.name in recorders_dict:
            raise ValueError(f"A recorder with the name '{recorder.name}' "
                             "is already attached to this optimizer.")
        recorders_dict[recorder.name] = recorder

    # TODO: Deprecate, when we stop storing recorders
    def remove_recorder(self, recorder: Union[Recorder, str]):
        recorders_dict = self._recorders
        recorder_name = recorder if isinstance(recorder, str) else recorder.name
        if recorder.name in self._recorders:
            return self._recorders.pop(recorder.name)
        elif recorder.name in self.diagnostic_recorders:
            return self.diagnostic_recorders.pop(recorder.name)
        else:
            warn(f"Attempted to remove recorder '{recorder.name}' from "
                 f"{type(self).__name}, but no Recorder of that name is "
                 "attached to this optimizer.")
    def remove_recorders(self):
        """Remove all recorders and return them as a dictionary."""
        recorders = {**self._recorders, **self.diagnostic_recorders}  # Makes shallow copies
        self._recorders.clear()
        self.diagnostic_recorders.clear()
        return recorders

# %% [markdown]
# ### Alternated algorithm

# %% [markdown]
# ::::{tabbed} Non-batched latents
#
# | Symbol | Identifier | Meaning |
# |--------|------------|---------|
# |$λ^θ$   | `λθ`  | Learning rate for parameters |
# |$λ^η$   | `λη`  | Learning rate for latents |
# |$K^θ_b$/$T^θ_b$ | `Kθb`/`Tθb` | Batch size for parameters (time bins/seconds) |
# |$K^θ_r$/$T^θ_r$ | `Kθr`/`Tθr` | Per-batch burning/relaxation time for parameters (time bins/seconds) |
# |$N^θ_b$ | `Nθb` | Number of batches to draw per pass for parameters |
# |$N^η_b$ | `Nθb` | Number of updates per pass for latents |
#
# Recall that $K$ is the latest time index in a dataset. See [sampling.ipynb](./sampling.ipynb#Sampling-batches) for the batch selection algorithm.
#
# 1. Initialize the latents.
# 2. Do one *pass*:
#    - **Initialize**
#      1. `data` $\leftarrow$ `select_segment(data, subject, T)`.
#      2. (**TODO** Initialize differential equation histories with current inferred initial conditions.)
#      3. Update hyperparameters.
#    - **Parameter updates**
#      1. Draw $N^θ_b$ start times for the batches:
#         $b^θ \leftarrow$ [`sample_batch_starts`$(K, K^θ_b + K^θ_r, N^θ_b)$](./sampling.ipynb#Sampling-batches)
#      5. Order the batches in **in**creasing order ($b^θ_1 < b^θ_2 < \dotsb$)
#      6. For $b^θ_s$ in $b^θ$:
#          1. Integrate the model up to $b^θ_s + K^θ_r + K^θ_b$.
#          2. Compute $g_θ$ on the segment $[b^θ_s+K^θ_r,b^θ+K^θ_r+K^θ_b]$ with Eq. [gθ](eq:gradient-params).
#          3. Update the parameters, e.g. with Adam.
#    - **Latent updates**
#      1. Finish integrating the model up to $K$.
#      2. Repeat $N^η_b$ times:
#          1. Compute $g_η'$ on the **entire** segment w.r.t $\step{s}{η}$ with Eq. [gη](eq:gradient-latents). (The vector $\step{s}{η}$ **includes** initial conditions).
#          2. Update the latents on the segment with $\step{s}{η} = \step{s-1}{η} + λ_η \step{s}{g_η}$.
#    - **Bookkeeping**
#      1. Increment `stepi`.
#      15. Invalidate stored histories for $k \geq 0$.
# 4. If not converged, return to 2.
#    If converged, exit.
#
# :::{note}
# $N^θ_{b}$ should be set to a relatively small number and be roughly independent of data length $K$.
# :::
#
# ::::

# %% [markdown] jupyter={"source_hidden": true}
# ::::{tabbed} Batched latents
#
# | Symbol | Identifier | Meaning |
# |--------|------------|---------|
# |$λ^θ$   | `λθ`  | Learning rate for parameters |
# |$λ^η$   | `λη`  | Learning rate for latents |
# |$K^θ_b$/$T^θ_b$ | `Kθb`/`Tθb` | Batch size for parameters (time bins/seconds) |
# |$K^θ_r$/$T^θ_r$ | `Kθr`/`Tθr` | Per-batch burning/relaxation time for parameters (time bins/seconds) |
# |$N^θ_b$ | `Nθb` | Number of batches to draw per pass for parameters |
# |$K^η_b$/$T^η_b$ | `Kηb`/`Tηb` | Batch size for latents (time bins/seconds) |
# |$K^η_r$/$T^η_r$ | `Kηr`/`Tηr` | Per-batch burning/relaxation time for latents (time bins/seconds) |
# |$N^η_b$ | `Nθb` | Number of batches to draw per pass for latents |
#
# Recall that $K$ is the latest time index in a dataset. See [sampling.ipynb](./sampling.ipynb#Sampling-batches) for the batch selection algorithm.
#
# 1. Initialize the latents.
# 2. Do one *pass*:
#    - **Initialize**
#      1. `data` $\leftarrow$ `select_segment(data, subject, T)`.
#      2. (**TODO** Initialize differential equation histories with current inferred initial conditions.)
#      3. Update hyperparameters.
#    - **Parameter updates**
#      1. Draw $N^θ_b$ start times for the batches:
#         $b^θ \leftarrow$ [`sample_batch_starts`$(K, K^θ_b + K^θ_r, N^θ_b)$](./sampling.ipynb#Sampling-batches)
#      5. Order the batches in **in**creasing order ($b^θ_1 < b^θ_2 < \dotsb$)
#      6. For $b^θ_s$ in $b^θ$:
#          1. Integrate the model up to $b^θ_s + K^θ_r + K^θ_b$.
#          2. Compute $g_θ$ on the segment $[b^θ_s+K^θ_r,b^θ+K^θ_r+K^θ_b]$ with Eq. [gθ](eq:gradient-params).
#          3. Update the parameters, e.g. with Adam.
#    - **Latent updates**
#      1. Finish integrating the model up to $K$.
#      8. Draw $N^η_b$ start times for the batches:
#      $b^η \leftarrow$ [`sample_batch_starts`$(K, K^η_b + K^η_r, N^η_b)$](./sampling.ipynb#Sampling-batches)
#      9. Order the batches in **de**creasing order ($b^η_1 > b^η_2 > \dotsb$)
#      10. With $b^η_s = b^η_{1}$:
#          1. Compute $g_η'$ on the segment $[b^η_s,b^η_s+K^η_b+K^η_r]$ w.r.t $\step{s}{η_{b^η_s:b^η_s+K^η_b+K^η_r}}$ with Eq. [gη](eq:gradient-latents) ($η$ **excludes** initial conditions. $g_η'$ **excludes** contributions from latent dynamics).
#          2. Update the latents on the segment with $\step{s}{η_{b^η_s:b^η_s+K^η_b+K^η_r}} = \step{s-1}{η_{b^η_s:b^η_s+K^η_b+K^η_r}} + λ_η \step{s}{g_η}$.
#      11. For $b^η_s$ in $\{b^η_2, \dotsc\, b^η_{N^η_b-1}\}$:
#          1. Compute $g_η$ on the segment $[b^η_s,b^η_s+K^η_b+K^η_r]$ w.r.t $\step{s}{η_{b^η_s:b^η_s+K^η_b}}$ with Eq. [gη](eq:gradient-latents) ($η$ **excludes** initial conditions).
#          2. Update the latents on the segment with $\step{s}{η_{b^η_s:b^η_s+K^η_b}} = \step{s-1}{η_{b^η_s:b^η_s+K^η_b}} + λ_η \step{s}{g_η}$.
#          3. (Optional) Invalidate stored histories for $k \geq b^η$.
#      12. With $b^η_s = b^η_{N^η_b}$:
#          1. Compute $g_η$ on the segment $[b^η_s,b^η_s+K^η_b+K^η_r]$ with Eq. [gη], w.r.t. $η_{:b^η_s+K^η_b}$ ($η$ **includes** initial conditions).
#          2. Update the latents on the segment with $\step{s}{η_{:b^η_s+K^η_b}} = \step{s-1}{η_{:b^η_s+K^η_b}} + λ_η \step{s}{g_η}$.
#    - **Bookkeeping**
#      1. Increment `stepi`.
#      15. Invalidate stored histories for $k \geq 0$.
# 4. If not converged, return to 2.
#    If converged, exit.
#
# :::{note}
# - $N^θ_{b}$ should be set to a relatively small number and be roughly independent of data length $K$; $N^η_b$ should be much larger since there are many more latent variables, and their gradients are easier to evaluate.
#
# - For the latent updates, the gradients on the earliest and latest batches are extended up to the edges of the data segment. This is to avoid effects where edges are left un-updated; points are undersampled around edges, and in particular, the last $Kηr$ time points would otherwise never be updated. Moreover, the initial condition has a disproportionate impact on the initial dynamics, so it makes sense to ensure it is updated on every pass.
#
# - A momentum-based update for latents is unlikely to work, since on each batch we update different latent variables.
# :::
#
# ::::

    # %%
    @add_to('AlternatedSGD')
    def draw_data_segment(self):
        """
        Draw the next data segment from `self.data_segment`.
        If it is different from the current (or if this is the first draw),
        the model histories are updated.
        If there is no current estimate for the latents for this segment,
        the model is integrated (with current parameters) to produce an
        initial estimate.

        **Side-effects**
        If the drawn segment is different from the current one:

        - Will model histories
        - Will update `_current_segment_key`

        Returns
        -------
        segmentkey: tuple
        """
        trialkey, trialslice, data = next(self.data_segments)
        segmentkey = trialkey + (trialslice.start, trialslice.stop, trialslice.step)
        current_segment_key = getattr(self, '_current_segment_key', None)
        if segmentkey != current_segment_key:
            # Changing the data segment requires:
            #   - Replacing all the data in the observed hists with that in the segment
            #   - Loading the current estimate of the latent history for that segment.
            # NB: We need to use some of the observed data to set the
            #     initial conditions (ICs) of the observed histories.
            #     This means that for histories which don't need ICs (or not
            #     as many), we must discard those time points.
            #     This also means that we can't integrate the model all
            #     the way to the end.
            max_pad_left = max(h.pad_left for h in self.observed_hists.values())
                # We need this much of the initial data points to initialize
            K_padded = self.model.time.Index.Delta(len(data.time))
                # The number of time bins we have
                # Some of these will be used to set ICs (i.e. the padding bins)
            K = K_padded - max_pad_left
                # The unpadded number of time bins
            # TODO: assert data.time.dt == self.model.time.dt
            # TODO: Match data.time to h.time ?
            for h in self.observed_hists.values():
                assert h.locked
                h.unlock()
                Δi = max_pad_left - h.pad_left
                h[:h.t0idx+K] = data[h.name][Δi:]
                    # NB: By using AxisIndex objects here, we make use of
                    # AxisIndex's validation to ensure each history uses
                    # the right padding and goes exactly up to K.
                    # For the difference between axis and data indices, see
                    # https://sinn.readthedocs.io/en/latest/axis.html#indexing
                h.lock(warn=False)
                # Remark: If we are setting latents to 'observed' (e.g. to fit
                # on ground truth), their initial cond is not returned by the
                # SyntheticDataAccessor, nor is it needed. In this case
                # Δi is equal to self.model.pad_left.

            # Observed histories may not have been set all the way to their end,
            # but they should all be in sync
            cur_tidx = h.cur_tidx
                # NB: We can use `cur_tidx` as an arg to `integrate`, because
                # it's an AxisIndex and thus will be appropriately shifted.
                # But it's numerical value depends on the amound of padding
                # on h.
            assert all(h.cur_tidx - h.t0idx + 1 == K
                       for h in self.observed_hists.values())

            ## Initialize model and set the latents ##
            # TODO: If we have multiple segments in the same trial, they should share a cache
            if segmentkey not in self.latent_cache:
                # Initialize the latents by integrating the model
                # This uses the current estimate of the parameters
                #was_locked = {}
                for h in self.latent_hists.values():
                    #was_locked[h] = h.locked
                    h.unlock()
                # TODO: What if initialize can take arguments ?
                # TODO: How do we set initial conditions ?
                # (Remark: these questions aren't crucial, since the latent,
                #  including its IC, will be inferred)
                self.model.clear()
                self.model.initialize()
                self.model.integrate(upto=cur_tidx, histories=self.latent_hists.values())
                for h in self.latent_hists.values():
                    #if was_locked[h]:
                    h.lock(warn=False)
                # Initialize the cache
                # This is not strictly necessary, but avoids not saving the
                # latents at all when `update_latents` is False.
                self.latent_cache[segmentkey] = utils.dataset_from_histories(
                    self.latent_hists.values())
                self.model.clear()

            else:
                latents = self.latent_cache[segmentkey]
                for h in self.latent_hists.values():
                    h.unlock()
                    #latent_data = self.get_latent_estimate(h, data)
                    h[:] = latent[h.name]
                    h.lock(warn=False)
                # TODO: initial conditions
                #for h,v in self.init_conds.items():
                #    was_locked = False
                #    if h.locked:
                #        was_locked = True
                #        h.unlock()
                #    h[:h.t0idx] = v
                #    if was_locked:
                #        h.lock(warn=False)

            object.__setattr__(self, '_current_segment_key', segmentkey)

        return segmentkey

    @add_to('AlternatedSGD')
    def step(self, update_params=True, update_latents=True):
        # NB: `status` must be updated by the optimization Task (c.f. OptimizeTask)
        if (self.status & OptimizerStatus.Converged) is OptimizerStatus.Converged:
            # Bitwise & will match both 'Converged' and 'Failed'
            logger.info(f"Skipping `step` because optimizer has status <{self.status}>.")
            return

        # Bind flag locally (minor optimization)
        diagnostic_hooks = config.diagnostic_hooks

        ## Clear the model ##
        # -> We do this here instead of the end, so that the final integrated model
        #    can be further inspected.
        for h in self.latent_hists.values():
            h.lock(warn=False)
        self.model.clear()

        ## Select data segment ##
        segmentkey = self.draw_data_segment()

        ## Update hyperparameters ##
        if self.update_hyperparams is not None:
            try:
                utils.recursive_set_value(self.fit_hyperparams,
                                          self.update_hyperparams(self))
            except TypeError as e:
                raise ValueError(
                    "The `update_hyperparams` callback must return"
                    "non-symbolic values.") from e

        ## Do updates ##
        # We need the data length (K) for `sample_batch_starts`
        # This length is typically less than unpadded_length, because of ICs
        # on observed histories (see `draw_data_segment`)
        # K = next(iter(self.observed_hists.values())).unpadded_length
        # assert all(K == h.unpadded_length for h in self.observed_hists.values())
        h = next(iter(self.observed_hists.values()))
        K = h.cur_tidx - h.t0idx + 1
        assert all(K == (h.cur_tidx - h.t0idx + 1) <= h.unpadded_length
                   for h in self.observed_hists.values())
        rng=self.rng

        #### Do parameter updates ####
        if update_params and not self.status & OptimizerStatus.ParamsConverged:
            Kθb=self.Kθb.eval(); Kθr=self.Kθr.eval();
            Nθb=self.Nθb;
            bθ = sample_batch_starts(K, Kθb+Kθr, Nθb, rng=rng)
            bθ.sort()
            for bθs in bθ:
                # FIXME: include Kθr in BPTT for later values
                self.model.integrate(upto=bθs+Kθr+Kθb-1)  # -1 because 'upto' is inclusive bound
                self.update_θ(bθs, Kθb, Kθr)
                diagnostic_hooks and self.record_diagnostics(bθs, 'θ', force=False)

        #### Do latent updates ####
        if update_latents and hasattr(self, 'update_η') and not self.status & OptimizerStatus.LatentsConverged:
            # Second condition because if there are no latent hists, we don't compile update_η
            Kηb=self.Kηb.eval(); Kηr=self.Kηr.eval();
            Nηb=self.Nηb;
            self.model.integrate(upto='end')
            if self.batched_latents:
                bη = sample_batch_starts(K, Kηb+Kηr, Nηb, rng=rng)
                bη.sort()
                self.update_η['rightmost'](bη[-1], Kηb, Kηr)
                    # args: start, batch length, derivative length
                diagnostic_hooks and self.record_diagnostics(bη[-1], 'η_rightmost', force=False)
                update_η = self.update_η['default']
                for bηs in bη[-2:0:-1]:  # Loop in reversed order, excluding first batch
                    diagnostic_hooks and self.record_diagnostics(bηs, 'η_default', force=False)
                    update_η(bηs, Kηb, Kηr)
                self.update_η['leftmost'](bη[0], Kηb, Kηr)
            else:
                # assert np.can_cast(self.model.time.unpadded_length, Kηb.dtype)
                # Kηb = self.model.time.unpadded_length.astype(Kηb.dtype)
                assert np.can_cast(K, Kηb.dtype)
                Kηb = K.astype(Kηb.dtype)
                update_η = self.update_η['default']
                for step in range(Nηb):
                    update_η(self.model.t0idx, Kηb, 0)
            diagnostic_hooks and self.record_diagnostics(bη[0], 'η_leftmost', force=False)

        ### Remove parameter degeneracies
        #if hasattr(self.model, 'remove_degeneracies'):
        #    self.model.remove_degeneracies(exclude=self.observed_hists)

        ##  Save latent value for next round  ##
        if update_latents:
            self.latent_cache[segmentkey] = utils.dataset_from_histories(
                self.latent_hists.values())

        ## Increment step number ##
        self.stepi += 1

        ## Store output from recorders ##
        # TODO: Will be deprecated
        self.record(force=False)

        # ## Perform convergence checks and update status ##
        # # TODO: Will be deprecated
        # for convergence_test in self.convergence_tests:
        #     self.status |= convergence_test(self)

    # TODO: Deprecate record
    @add_to('AlternatedSGD')
    def record(self, force=True):
        """It is presumed that if a user uses `record` in their own code, they
        expect it to record unconditionnally. There the default is `force=True`
        """
        for recorder in self._recorders.values():
            if force or recorder.ready(self.stepi):
                recorder.record(self.stepi, self)

    # TODO?: Also deprecate record_diagnostics ?
    @add_to('AlternatedSGD')
    def record_diagnostics(self, batch_start, context, force=True):
        for recorder in self.diagnostic_recorders.values():
            if force or recorder.ready(self.stepi, batch_start, context):
                recorder.record(self.stepi, self, batch_start, context)

# %% [markdown]
# ### Compilation of the optimization functions
#
# In the steps below, “locking” a history indicates to *sinn* that it should be treated as data. This means that when constructing an accumulator, that history's update function is not called, even if it is part of the state histories. It is an error to index into a history beyond its current time index. \
# Filling a history with data has a similar effect: if it is unlocked, *sinn* will attempt to evaluate its update function, but abort once it sees that data is already available. Nevertheless, this is fragile and may still result in unterminated recursion, which is way the locking mechanism exists. It is an error to modify a locked history, so filling should be done first. \
# In general, use *lock* to indicate that a history is to be treated as data, and fill it with data to prevent indexing errors.
#
# The compilation procedure is split into 3 steps:
# - [A preparatory step](#Compilation-–-Preparatory-step) (setting up variables, ensuring histories are locked & fill as required).
# - [Compilation of the optimizer update function for parameters ($θ$)](#Compilation-of-the-parameter-update-function).
# - [Compilation of the optimizer update function for latent histories ($η$)](#Compilation-of-the-latent-update-function).
#
# Each compiled function both computes the appropriate gradient, and updates the values of the parameters or latent values accordingly.

    # %%
    @add_to('AlternatedSGD')
    def compile_optimization_functions(
        self,
        initializer: Optional[str]=None,
        force_initialize: bool=False,
        clear: bool=True
        ):
        """
        .. Note:: The compilation routine needs to initialize the model, but
           if your model has a default initialization, it's generally not
           necessarry to specify the `initializer`.
           You can always re-initialize the model before training.

        Parameters
        ----------
        initializer: str | None
            Value to pass to `model.initialize`
        force_initialize: bool
            False (default): Only call `model.initialize` if some histories require it
                (i.e. their curtidx is less than t0-1).
            True: Always call `model.initialize`.
        clear: bool
            Ignored if `force_initialize` is ``True``.
            True (default): Clear all non-observed histories. Specifically:
                - Lock all observed histories.
                - Unlock all observed histories.
                - Call `model.clear`.
            False: Do none of the above. The locked status of histories is
                left unchanged.

        **Side-effects**
        - Sets model parameters to match current optimization paramters (`self.Θ`)
        - May clear model histories and change their locked status, as described
        above.

        """
        # This is only really required when calling `initialize`, but for
        # consistency we do it all the time.
        self.model.params.set_values(
            self.prior_params.backward_transform_params(self.Θ.get_values()),
            must_not_set_other_params=False  # Required because at present, composite models use generic SubmodelParams for subparam sets
        )
        ## Initialize model ##
        if force_initialize or any(h.cur_tidx < h.t0idx-1 for h in self.model.history_set):
            # uninitialized_hists = [f"  {nm} – cur tidx: {h.cur_tidx}, t0idx: {h.t0idx}"
            #                        for nm, h in self.model.nested_histories.items()
            #                        if h.cur_tidx < h.t0idx-1]
            # if uninitialized_hists:
            #     uninit_line = (
            #         "\nThis also means that, instead of relying on the model's "
            #         "`initialize` method, all histories must already be initialized. "
            #         "(I.e. their current time index should be at least "
            #         "`t0idx-1`.) This is is not the case for the following "
            #         "histories:\n" + "\n".join(uninitialized_hists) )
            # else:
            #     uninit_line = ""
            # raise NotImplementedError(
            #     "Because we can't reliably transform model parameters to optim "
            #     "parameters, it is not currently possible to initialize "
            #     "the fit from the model parameters." + uninit_line)
            self.model.initialize(initializer)
            # self.Θ.set_values(self.prior_params.forward_transform_params(self.model.params.get_values()))
                # FIXME: Above won't work if there are deterministics in the prior
        elif clear:
            for h in self.model.history_set:
                if h in self.observed_hists.values():
                    h.lock(warn=False)
                else:
                    h.unlock()
            self.model.clear()

        ## Run the compilation functions ##
        self.prepare_optimizer_compilation()
        self.compile_parameter_optimizer()
        if self.latent_hists.values():
            self.compile_latent_optimizers()
        self.cleanup_optimizer_compilation()

        ## Mark optimizer as compiled ##
        object.__setattr__(self, 'compiled', True)

# %% [markdown]
# #### Compilation – Preparatory step
#
# 1. Create placeholder shared variables for $K^θ_b$, $K^θ_r$, $K^η_b$ and $K^η_r$.
#    - These are required because histories can only be indexed by shared variables (not pure symbolics). The placeholder variables are replaced by symbolic ones before compilation.
# 1. Initialize the model.
#    - This sets the initial conditions.
# 2. Fill the history corresponding to observations with (possibly dummy) data. \
#    (This data may be changed after compilation.)
# 3. Integrate the model one time point.
#    - The creation of the latent updates function needs at least one time point where all model histories are already calculated.
# 4. Lock the histories corresponding to observations.

    # %%
    @add_to('AlternatedSGD')
    def prepare_optimizer_compilation(self):
        """
        Do the following:

        * If required, fill observed histories with dummy data.
        * If required, integrate one time point.
        * Lock observed histories.
        * Create required the shared & symbolic variables
        """

        object.__setattr__(self, '_compile_context', SimpleNamespace())
        object.__setattr__(self, '_k_vars', SimpleNamespace())

        ## Ensure that all latent hists are synchronized with model cur_tidx ##
        model_tidx = self.model.cur_tidx
        model_tnidx = self.model.tnidx
        for h in self.latent_hists.values():
            # FIXME?: Can't use != because it compares unequal for indices from different axes
            #         I think this is required because the != test is used to determine if an
            #         AxisIndex needs to be converted when doing comparisons/arithmetic.
            #         Still, it's rather unintutive.
            if h.cur_tidx > model_tidx or h.cur_tidx < model_tidx:
                # < and > comparators account for padding,
                raise RuntimeError(
                    "AlternatedSGD: Latent history is not synchronized with the model.\n"
                    f"History {h.name} curtidx: {h.cur_tidx}\n"
                    f"Model {self.model.name} curtidx: {model_tidx}")

        ## Ensure intermediate histories are alse synchronized ##
        # Any history neither latent nor observed should only be required for
        # intermediate calculations and therefore unlocked.
        # It also should not be computed further than any latent hist, since
        # it could prevent a latent hist from being computed.
        # NB: self.model.nested_histories returns duplicates, if a history is
        #     part of more than one submodel. We use id(h) to ensure that
        #     a) we only keep one copy in intermediate_hists
        #     b) any copy is recognized as part of latent_or_observed_hists, if applicable
        latent_or_observed_hists = set(id(h) for h in self.latent_hists.values()) \
                                   | set(id(h) for h in self.observed_hists.values())
        intermediate_hists = {id(h): h for h in self.model.nested_histories.values()
                              if id(h) not in latent_or_observed_hists}
        for h in intermediate_hists.values():
            if h.locked:
                raise RuntimeError(
                    f"AlternatedSGD: History {h.name} is locked, but not listed as observed.")
            if h.cur_tidx > model_tidx:
                raise RuntimeError(
                    "AlternatedSGD: Intermediate history is not synchronized with the model.\n"
                    f"History {h.name} curtidx: {h.cur_tidx}\n"
                    f"Model {self.model.name} curtidx: {model_tidx}")

        ## Mark the lock state of each observed history ##
        lock_states = {h: h.locked for h in self.observed_hists.values()}
        self._compile_context.lock_states = lock_states

        ## Fill with dummy data ##
        # Temporarily fill the observed histories with dummy data; we only need data for 2 time points
        reset_curtidcs = {h: h.cur_tidx for h in self.observed_hists.values()}  # Keep the curtidx so we can invalidate the dummy data once we are done.
        for h, kh in reset_curtidcs.copy().items():
            if kh < h.t0idx+2:
                h.unlock()
                h[kh+1:h.t0idx+3] = 0
                if lock_states[h]: h.lock(warn=False)
            else:
                # No dummy data -> no need to invalidate -> remove from reset list
                del reset_curtidcs[h]
        self._compile_context.reset_curtidcs = reset_curtidcs

        ## Ensure there is at least one uncomputed point (required by model.accumulate) ##
        if model_tidx == model_tnidx:
            # Model is already filled to the end
            # Using : in index catches any right-padding
            # FIXME: Don't use `shim.eval` (expensive compilation). Essentially we need something like
            # `.data`, but that keeps the padding and does AxisIndex conversion
            deleted_data = {h: shim.eval(h[model_tnidx:])
                            for h in self.model.history_set
                            if h.cur_tidx >= model_tidx}
            self._compile_context.deleted_data = deleted_data
            self._compile_context.deleted_tidx = model_tnidx
            with unlocked_hists(*self.model.history_set):
                self.model.clear(after=model_tnidx-1)
        else:
            self._compile_context.deleted_data = {}
            self._compile_context.deleted_tidx = None
        assert self.model.cur_tidx < self.model.tnidx

        ## Integrate one time point ##
        if self.model.cur_tidx < self.model.t0idx:
            self.model.integrate(upto=0)  # FIXME: Shouldn't this be upto=model.t0idx ?

        ## Lock observed histories ##
        for h in self.observed_hists.values():
            if not h.locked:
                h.lock(warn=False)

        ## Normalize fit hyperparameters ##
        hyperθ = self.fit_hyperparams
        # Ensure that the latent learning rate is a dict
        # IMPORTANT: JSON only allows plain args as keys, so we use `h.name`
        if not isinstance(hyperθ['latents']['λη'], dict):
            hyperθ['latents']['λη'] = {hname: hyperθ['latents']['λη']
                                       for hname, h in self.latent_hists.items()}
        else:
            if not all(hname in hyperθ['latents']['λη'] for hname in self.latent_hists):
                raise AssertionError("Not all latent histories are listed in latent hyperparameters.")
        # Broadcast all latent learning rates to the shape of the corresponding history
        λη = hyperθ['latents']['λη']
        for hnm, ληh in λη.items():
            h = getattr(self.model, hnm)
            λη[hnm] = np.broadcast_to(ληh, h.shape)
        # Convert the hyperparameters to shared variables
        for cat in ['params', 'latents']:
            for nm, val in hyperθ[cat].items():
                if isinstance(val, dict):
                    hyperθ[cat][nm] = {  # getattr is to deal with histories
                        # varnm: shim.shared(val, name=f"{nm} ({getattr(varnm, 'name', varnm)})")
                        varnm: shim.shared(val, name=f"{nm} ({varnm})")
                        for varnm, val in val.items()
                    }
                else:
                    hyperθ[cat][nm] = shim.shared(val, name=nm)

        ## Create placeholder batch size variables ##
        # Do this after integration, filling & locking, to ensure histories are synchronized
        model = self.model
        modelstr = f"({type(model).__name__})"
        self._k_vars.k = model.time.Index(model.num_tidx)  # -> curtidx_var
        self._k_vars.Kθb = model.time.index_interval(self.fit_hyperparams['Tθb']*model.time.unit)
        self._k_vars.Kθr = model.time.index_interval(self.fit_hyperparams['Tθr']*model.time.unit)
        self._k_vars.Kθb = model.time.Index.Delta(shim.shared(self.Kθb.plain, name=f"Kθb ({modelstr})"))  # -> Kθb_symb
        self._k_vars.Kθr = model.time.Index.Delta(shim.shared(self.Kθr.plain, name=f"Kθr ({modelstr})"))  # -> Kθr_symb
        self._k_vars.Kθb_symb = shim.tensor(self.Kθb.plain)  # Symbolic variable associated to a shared variable
        self._k_vars.Kθr_symb = shim.tensor(self.Kθr.plain)
        self._k_vars.Kηb = model.time.index_interval(self.fit_hyperparams['Tηb']*model.time.unit)
        self._k_vars.Kηr = model.time.index_interval(self.fit_hyperparams['Tηr']*model.time.unit)
        self._k_vars.Kηb = model.time.Index.Delta(shim.shared(self.Kηb.plain, name=f"Kηb ({modelstr})"))  # -> Kηb_symb
        self._k_vars.Kηr = model.time.Index.Delta(shim.shared(self.Kηr.plain, name=f"Kηr ({modelstr})"))  # -> Kηr_symb
        self._k_vars.Kηb_symb = shim.tensor(self.Kηb.plain)
        self._k_vars.Kηr_symb = shim.tensor(self.Kηr.plain)

    @add_property_to('AlternatedSGD')
    def k(self):
        return self._k_vars.k
    @add_property_to('AlternatedSGD')
    def Kθb(self):
        return self._k_vars.Kθb
    @add_property_to('AlternatedSGD')
    def Kθr(self):
        return self._k_vars.Kθr
    @add_property_to('AlternatedSGD')
    def Kθb_symb(self):
        return self._k_vars.Kθb_symb
    @add_property_to('AlternatedSGD')
    def Kθr_symb(self):
        return self._k_vars.Kθr_symb
    @add_property_to('AlternatedSGD')
    def Kηb(self):
        return self._k_vars.Kηb
    @add_property_to('AlternatedSGD')
    def Kηr(self):
        return self._k_vars.Kηr
    @add_property_to('AlternatedSGD')
    def Kηb_symb(self):
        return self._k_vars.Kηb_symb
    @add_property_to('AlternatedSGD')
    def Kηr_symb(self):
        return self._k_vars.Kηr_symb
    @add_property_to('AlternatedSGD')
    def Nθb(self):
        return self.fit_hyperparams['Nθb']
    @add_property_to('AlternatedSGD')
    def Nηb(self):
        return self.fit_hyperparams['Nηb']

    @add_to('AlternatedSGD')
    def cleanup_optimizer_compilation(self):
        """
        Do the following:

        Undo the actions of `prepare_optimizer_compilation`:
            - Clear dummy data
            - Reinsert deleted data
        """
        del_tidx = self._compile_context.deleted_tidx
        if del_tidx is not None:
            assert self.model.cur_tidx < del_tidx
            for h, data in self._compile_context.deleted_data.items():
                assert h._sym_tidx is h._num_tidx
                with unlocked_hists(h):
                    h[del_tidx:] = data
            assert self.model.cur_tidx == del_tidx

        for h, kh in self._compile_context.reset_curtidcs.items():
            with unlocked_hists(h):
                h.clear(after=kh)

        # Return locked histories to their state before compilation
        for h, locked in self._compile_context.lock_states.items():
            if not locked:
                h.unlock(warn=False)

        if not config.diagnostic_hooks:
            # Keep _compile_context for inspection in a notebook
            object.__delattr__(self, '_compile_context')

# %% [markdown]
# #### Compilation of the parameter update function
#
#    1. Lock the histories corresponding to latent variables.
#    2. Evaluate `forward_logp` with symbolic batch start and length variables.
#       - This creates an unanchored $l_{k:k+K^θ_b+K^θ_r}$ expression that can be evaluated at any start time $k$.
#       - This also returns a set of state updates, which we can use to integrate the model forward from $k$ to $k+K^θ_b+K^θ_r$.
#    4. Use *Adam* to compute updates for the model parameters.
#    5. Compile a function which both updates the parameters and integrates the model from $k$ to $k+K^θ_b+K^θ_r$.
#    6. Flush the global symbolic updates.
#       - These were created when evaluating `forward_logp`, and should be kept until the final function is compiled.
#
# Compiled function: ``AlternatedSGD.update_θ``$(k, K^θ_b, K^θ_r)$.

    # %%
    ## Compile the parameter update function ##
    @add_to('AlternatedSGD')
    def compile_parameter_optimizer(self):

        if shim.pending_updates():
            raise AssertionError("There are pending symbolic updates.")
        Kθb_symb = self.Kθb_symb
        Kθr_symb = self.Kθr_symb

        #### Lock latent histories ####
        for h in self.latent_hists.values():
            h.lock(warn=False)

        #### Sanity check - differentiable update ####
        if self.model.rng_hists:
            raise RuntimeError(
                 "There are unlocked histories which depend on a random "
                 "number generator: this is almost always a mistake, since "
                 f"the parameter logp function `{self.logp_params.__qualname__}` "
                 "will not be differentiable.\nConsider locking these "
                 "histories, either by providing observation data, or by "
                 "listing them among the latent histories.\n"
                 f"Histories with RNG inputs: {[h.name for h in self.model.rng_hists]}."
            )

        #### Evaluate logp(θ) ####
        batch_logp, state_upds = self.logp_params(self.model.curtidx_var,
                                                  Kθb_symb)
        # Convert from model vars to optim vars
        batch_logp = self.prior_params.sub_optim_vars(batch_logp, self.Θ)
        state_upds = self.prior_params.sub_optim_vars(state_upds, self.Θ)
            # Keys to state_upds are histories, so only values may need to be transformed
        # Add prior. The factor λ scales the prior proportionally to the size of a batch
        λ = self._k_vars.Kθb_symb / len(self.model.time)
        batch_logp += λ * self.prior_params.sub_optim_vars(self.prior_params.logpt, self.Θ)
        #if self.logp_params_regularizer is not None:
        #    batch_logp += self.logp_params_regularizer(self.model)
        if state_upds != shim.get_updates():
            raise AssertionError("State updates differ from shim updates.")
        # There should be no latents in the updates => they need to be integrated first
        latent_data = [h._num_data for h in self.latent_hists.values()]
        if any(v in latent_data for v in state_upds):
            raise AssertionError("Some of the state updates are for latent variables.")

        #### Retrieve parameters to optimize (WIP) ####
        #Θ = [θ for θ in self.prior_params.free_RVs
        #     if not isinstance(getattr(θ, 'distribution', None), pm.Constant)]
        #symb_inputs = {θ.name: θ for θ in shim.graph.symbolic_inputs(batch_logp)}
        #Θ = [symb_inputs[θname] for θname in self.params]
        #if len(Θ) != len(self.params):
        #    raise shim.graph.MissingInputError(
        #        f"{self.__qualname__} was asked to optimize the following "
        #        "parameters, but they don't appear in the graph of the loss: "
        #        f"{set(self.params) - set(symb_inputs)}.")

        #### Compute parameter updates with Adam ####
        if self.param_optimizer is not None:
            param_upds = self.param_optimizer(
                -batch_logp, params=[θ for _, θ in self.Θ], **self.fit_hyperparams['params'])
        else:
            param_upds = self.default_param_optimizer(
                -batch_logp, params=[θ for _, θ in self.Θ], **self.fit_hyperparams['params'])
        if state_upds != shim.get_updates():
            new_upds  = [v for v in shim.get_updates() if v not in state_upds]
            chgd_upds = [v for v in state_upds if state_upds[v] != shim.get_updates()[v]]
            raise RuntimeError(f"The param optimizer `{self.param_optimizer.__qualname__}` "
                               "should return an update dictionary, but must "
                               "not modify `shim`'s global update dictionary.\n"
                               f"Added or modified variables: {new_upds+chgd_upds}.")

        #### Compile logp function ####
        logp_f = shim.graph.compile(
            inputs =(self.model.curtidx_var, Kθb_symb),
            outputs=batch_logp
        )
        def logp(k0=self.model.time.t0idx,K=len(self.model.time)):
            # Typically the forward accumulator is offset by 1; shift k0 accordingly
            if not hasattr(self.logp_params, 'start_offset'):
                raise AttributeError(
                    "The `logp_params` argument to AlternatedSGD does not seem "
                    "to have been created with the `sinn.models.Model.accumulate` "
                    "decorator.")
            k0 -= self.logp_params.start_offset
            return logp_f(k0, K)
        object.__setattr__(self, 'logp', logp)

        #### Compile θ update function ####
        update_θ = shim.graph.compile(
            inputs =(self.model.curtidx_var, Kθb_symb, Kθr_symb),
            outputs=[],
            updates=param_upds,
            on_unused_input='ignore'
        )
        object.__setattr__(self, 'update_θ', update_θ)

        # In a notebook, keep the updates for later inspection
        if config.diagnostic_hooks:
            self._compile_context.param_batch_logp = batch_logp
            self._compile_context.state_upds = state_upds
            self._compile_context.param_upds = param_upds

        #### Flush symbolic updates ####
        self.model.theano_reset()

    @add_to('AlternatedSGD')
    def default_param_optimizer(self, cost, params, λθ, **kwargs):
        """
        Simply calls the mackelab_toolbox.optimizers.Adam optimizer.

        Parameters
        ----------
        params:
            Passed as-is to Adam.
        λθ:
            Passed as `lr` to Adam (learning rate).
        **kwargs:
            Passed on to Adam
        """
        return mtb.optimizers.Adam(cost, params=params, lr=λθ, **kwargs)

# %% [markdown]
# #### Compilation of the latent update function
#
#    1. Unlock the histories corresponding to latent variables.
#    2. Evaluate `backward_logp` with symbolic batch start and length variables.
#       - This again creates an unanchored $l_{k:k+K^θ_b+K^θ_r}$ expression.
#       - This time we discard the state updates, since we won't be integrating the model along with the latent updates.
#    3. Compute the symbolic gradient with respect to the latent variables.
#       - It is in fact cheaper to compute the symbolic gradient of $l_{k:k+K^θ_b+K^θ_r}$ with respect to the *entire* latent trace(s). As long as we slice out the appropriate window from the result before compiling, the unused components will never be computed.
#    4. Create the three latent update rules (the right and leftmost gradients being extended to $k+K^η_b+K^η_r$ (rightmost) and 0 (leftmost)).
#       - These are obtained by incrementing the latent by $λ_η g_η$.
#    5. In the update rules, replace placeholder shared variables by symbolic ones, which can be used as function arguments.
#    6. Compile the three update functions.
#    7. As before, finish by flushing the global symbolic updates.
#
# Compiled function: ``AlternatedSGD.update_η``$(k, K^η_b, K^η_r)$.

    # %%
    ## Compile latent update function ##
    @add_to('AlternatedSGD')
    def compile_latent_optimizers(self):

        if shim.pending_updates():
            raise AssertionError("There are pending symbolic updates.")
        Kηb_symb = self.Kηb_symb
        Kηr_symb = self.Kηr_symb

        #### Unlock latent histories ####
        for h in self.latent_hists.values():
            h.unlock()

        #### Evaluate η (latent) updates ####
        if self.batched_latents:
            # We don't call 'theano_reset' after constructing each cost graph,
            # (maybe we should ?), but we do ensure the list of updates is empty.
            # This is to avoid updates from the first cost showing up in later ones.
            batch_logp, _ = self.logp_latents(
                self.model.curtidx_var, Kηb_symb+Kηr_symb)
            assert not shim.get_updates()
            # Initial batch depends on non-symbolic start => compute graph with
            # symbolic end points, then substitute the required values.
            initial_batch_logp, _ = self.logp_latents_nodyn(
                self.model.curtidx_var, self.model.batchsize_var)
            initial_batch_logp = shim.graph.clone(
                initial_batch_logp,
                replace={self.model.curtidx_var: self.model.t0idx,
                         self.model.batchsize_var: self.model.curtidx_var+Kηb_symb+Kηr_symb})
            assert not shim.get_updates()
            # Cost for the rightmost batch should not include latent dynamics, otherwise fit is unstable
            nodyn_batch_logp, _ = self.logp_latents_nodyn(
                self.model.curtidx_var, Kηb_symb+Kηr_symb)
            assert not shim.get_updates()

            batch_logp         = self.prior_params.sub_optim_vars(batch_logp, self.Θ)
            initial_batch_logp = self.prior_params.sub_optim_vars(initial_batch_logp, self.Θ)
            nodyn_batch_logp   = self.prior_params.sub_optim_vars(nodyn_batch_logp, self.Θ)

            #### Compute latent updates (see below) ####
            if self.latent_optimizer is not None:
                latent_updates = self.latent_optimizer(
                    -batch_logp, -initial_batch_logp, -nodyn_batch_logp, **self.fit_hyperparams['latents'])
            else:
                latent_updates = self.default_latent_optimizer(
                    -batch_logp, -initial_batch_logp, -nodyn_batch_logp, **self.fit_hyperparams['latents'])

        else:
            # The entire latent trace is updated each time, which means
            # that we don't have to define special cost graphs for the end points.
            # This is most appropriate for shorter traces.

            # As with the initial batch above, we compute the graph with
            # symbolic end points, then substitute required values.
            latent_logp, _ = self.logp_latents(self.model.curtidx_var, self.model.batchsize_var)
            assert np.can_cast(self.model.time.unpadded_length, self.model.batchsize_var.dtype)
            latent_logp = shim.graph.clone(
                latent_logp,
                replace={self.model.curtidx_var: self.model.t0idx,
                         self.model.batchsize_var: self.model.time.unpadded_length.astype(self.model.batchsize_var.dtype)}
            )
            assert not shim.get_updates()

            latent_logp = self.prior_params.sub_optim_vars(latent_logp, self.Θ)

            #### Compute latent updates (see below) ####
            if self.latent_optimizer is not None:
                latent_updates = self.latent_optimizer(
                    -latent_logp, None, None, **self.fit_hyperparams['latents'])
            else:
                latent_updates = self.full_trace_latent_optimizer(
                    -latent_logp, None, None, **self.fit_hyperparams['latents'])

        # In a notebook, keep a handle to the latent_updates for tests & inspection
        if config.diagnostic_hooks:
            self._compile_context.latent_batch_logp = batch_logp
            self._compile_context.latent_updates = latent_updates

        #### Compile the update functions ####
        update_η = {
            upd_type: shim.graph.compile(
                inputs = (self.model.curtidx_var, Kηb_symb, Kηr_symb),
                outputs = [],
                updates = updates,
                on_unused_input = 'ignore'
            )
            for upd_type, updates in latent_updates.items()
        }
        object.__setattr__(self, 'update_η', update_η)

        #### Flush global symbolic updates ####
        self.model.theano_reset()

    @add_to('AlternatedSGD')
    def default_latent_optimizer(self, cost, cost_with_init, cost_without_dyn,
                                 λη: Union[FloatX,dict], clip: Optional[float]=None):
        """
        Plain SGD (learning rate * gradient).

        Parameters
        ----------
        cost: Expression graph for the cost
        cost_with_init: Expression for the cost, including initial conditions
        cost_without_dyn: Expression graph for an alternate cost, which includes
            no contribution from latent dynamics.
            This is used for the rightmost batch, to improve fit stability.
        λη: Learning rate
            float: Same learning rate applied to all latents.
            dict: (hist name: learning rate) pairs. May include an extra entry
                  keyed with the string 'default'.
        clip: Maximum value of gradients.
            Gradients are clipped using an L∞ norm, so the direction is conserved.
            Specifically, each gradient component of each latent history is
            independently divided by `clip`; the largest of these ratios, if it
            exceeds 1, is used to rescale the whole gradient.

        Returns
        -------
        Dictionary with three entries. Each entry is a dictionary of update graphs,
        keyed by the history:
            {
             'default': {hist: hist_update, ...},
             'rightmost': {hist: hist_update, ...},
             'leftmost': {hist: hist_update, ...}
            }
        """

        latent_data = [h._num_data for h in self.latent_hists.values()]
        k   = self.k
        Kηb = self.Kηb
        Kηr = self.Kηr
        Kηb_symb = self.Kηb_symb
        Kηr_symb = self.Kηr_symb

        #### Compute gradients wrt latent variables ####
        gη = shim.grad(cost, latent_data) # grad returns a list
        init_gη = shim.grad(cost_with_init, latent_data) # grad returns a list
        nodyn_gη = shim.grad(cost_without_dyn, latent_data) # grad returns a list
        if clip:
            gη = clip_gradients(gη, clip)
            init_gη = clip_gradients(init_gη, clip)
            nodyn_gη = clip_gradients(nodyn_gη, clip)
        gη = {h: gηh for h,gηh in zip(self.latent_hists.values(), gη)}
        init_gη = {h: gηh for h,gηh in zip(self.latent_hists.values(), init_gη)}
        nodyn_gη = {h: gηh for h,gηh in zip(self.latent_hists.values(), nodyn_gη)}

        #### Create the slices ####
        #k_indices = {h: shim.print(k.convert(h.time).data_index, message="k (data index)")  # DEBUG
        #                + shim.print(k, message="k (anchor index)") - k for h in self.latent_hists.values()}  # DEBUG
        k_indices = {h: k.convert(h.time).data_index for h in self.latent_hists.values()}
            # We need to convert to .data_index ourselves, because we index directly into the underlying _num_data
        K_slices = {
            'default'  : {h: slice(kh, kh+Kηb.plain) for h,kh in k_indices.items()},
            'rightmost': {h: slice(kh, kh+Kηb.plain+Kηr.plain) for h,kh in k_indices.items()},
            #'rightmost': {h: slice(shim.print(kh, message=f"kh ({h.name})"), kh+Kηb.plain+Kηr.plain) for h,kh in k_indices.items()},  # DEBUG
            'leftmost' : {h: slice(None, kh+Kηb.plain) for h,kh in k_indices.items()}
        }

        # In a notebook, keep grad and K_slices for instrospection
        if config.diagnostic_hooks:
            self._compile_context.gη = {'default': gη, 'rightmost': nodyn_gη, 'leftmost': init_gη}
            self._compile_context.K_slices = K_slices

        #### Standardize the learning rate format ####
        if not isinstance(λη, dict):
            λη = {h: λη for h in self.latent_hists.values()}
        else:
            # Replace history names by the histories themselves
            λη_names = λη
            λη = {}
            for h in self.latent_hists.values():
                try:
                    λη[h] = λη_names[h.name]
                except KeyError:
                    try:
                        λη[h] = λη['default']
                    except KeyError:
                        raise ValueError("No learning rate specified for "
                                         f"history '{h.name}'.")

        #### Create the update dictionaries ####
        latent_updates = {}

        latent_updates['default'] = {
            h._num_data: shim.inc_subtensor(h._num_data[slc], -λη[h]*gη[h][slc])
            for h, slc in K_slices['default'].items()
        }

        latent_updates['rightmost'] = {}
        for h, slc in K_slices['rightmost'].items():
            Δt = getattr(h.dt, 'magnitude', h.dt)
            # slc = shim.print(slc, message="rightmost slice")  # DEBUG
            inc = -λη[h]*nodyn_gη[h][slc]
            # # The rightmost increment is rescaled by Δt, to correct for edge effects
            # #inc = shim.set_subtensor(inc[-1], shim.print(Δt*inc[-1], message="Δη_T"))  # DEBUG
            # inc = shim.set_subtensor(inc[-1], Δt*inc[-1])
            # #latent_updates['rightmost'][h._num_data] = shim.inc_subtensor(
            # #    shim.print_array(h._num_data, np.s_[-3:], message="I_T (before)")[slc], inc)  # DEBUG
            latent_updates['rightmost'][h._num_data] = shim.inc_subtensor(
                h._num_data[slc], inc)
            #latent_updates['rightmost'][h._num_data] = shim.print_array(
            #    latent_updates['rightmost'][h._num_data], np.s_[-3:], message="I_T (after)")  # DEBUG

        latent_updates['leftmost'] = {
            h._num_data: shim.inc_subtensor(h._num_data[slc], -λη[h]*init_gη[h][slc])
            for h, slc in K_slices['leftmost'].items()
        }

        #### Replace placeholder variables by symbolic ones ####
        for upd_dict in latent_updates.values():
            for num_data, upd in upd_dict.items():
                upd_dict[num_data] = shim.graph.clone(
                    upd, replace={k.plain:   self.model.curtidx_var,
                                  Kηb.plain: Kηb_symb,
                                  Kηr.plain: Kηr_symb}
                                  #self.model.batchsize_var: Kηb_sym+Kηr_symb})
                )

        return latent_updates

    @add_to('AlternatedSGD')
    def full_trace_latent_optimizer(self, cost, cost_with_init, cost_without_dyn,
                                    λη: Union[FloatX,dict], clip: Optional[float]=None):

        if cost_with_init or cost_without_dyn:
            raise TypeError("`full_trace_latent_optimizer()` requires only one cost; "
                            "`cost_with_init` and `cost_without_dyn` should be 0 or `None`, "
                            f"but they are respectively '{cost_with_init}' and '{cost_without_dyn}'.")

        latent_data = [h._num_data for h in self.latent_hists.values()]

        #### Compute gradients wrt latent variables ####
        gη = shim.grad(cost, latent_data) # grad returns a list
        if clip:
            gη = clip_gradients(gη, clip)
        gη = {h: gηh for h,gηh in zip(self.latent_hists.values(), gη)}

        # In a notebook, keep grad and K_slices for instrospection
        if config.diagnostic_hooks:
            self._compile_context.gη = gη

        #### Standardize the learning rate format ####
        if not isinstance(λη, dict):
            λη = {h: λη for h in self.latent_hists.values()}
        else:
            # Replace history names by the histories themselves
            λη_names = λη
            λη = {}
            for hname, h in self.latent_hists.items():
                try:
                    λη[h] = λη_names[hname]
                except KeyError:
                    try:
                        λη[h] = λη['default']
                    except KeyError:
                        raise ValueError("No learning rate specified for "
                                         f"history '{hname}'.")

        #### Create the update dictionaries ####
        latent_updates = {}

        latent_updates['default'] = {
            h._num_data: h._num_data - λη[h]*gη[h]
            for h in λη
        }

        return latent_updates

# %% tags=["remove-cell"]
AlternatedSGD.update_forward_refs()
