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
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # Base classes and definitions for optimizers

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["hide-input"]
import abc
import copy
from collections import namedtuple
from types import SimpleNamespace
from typing import Optional, Union, List, Dict, Callable
from enum import Flag
from numbers import Integral
from pydantic import BaseModel, validator, root_validator

import theano_shim as shim
from mackelab_toolbox.typing import json_like, Array, FloatX, RNGenerator
from sinn.histories import History
from sinn.models import ModelParams, initializer

from sinnfull._json_encoders import json_encoders
from sinnfull.parameters import ParameterSet
from sinnfull.models import Model
from sinnfull.sampling import SegmentSampler

# %%
__all__ = ["Optimizer", "OptimParams", "OptimizerStatus"]


# %% [markdown]
# ## Notation conventions
#
# - Uppercase $\Theta$ / `Θ`: set of parameters
# - Lowercase $\theta$ / `θ`: parameter, i.e. $\theta \in \Theta$
# - Lowercase $\eta$ / `η`: latent parameter(s)
# - $K$: Length in time bins
# - $T$: Length in continuous time

# %% [markdown]
# ## Specifying the optimization parameters
#
# The set of parameters to optimize are specified with a PyMC3 prior. This allows a great deal of flexibility, in particular the possibilities to transform the model variables to a more numerically advantageous space, and to provide complex prior distributions to help regularize training. By “PyMC3 prior”, we mean a PyMC3 model with no observed variables; see [Priors](priors).
#
# Parameters provided to an optimizer must correspond to the [_optimization space_](prior-types-and-spaces) (as opposed to the _model space_). The provided `OptimParams` allows to indicate this semantically, but it is up to you to provide the correct parameters.
#
# :::{Note}
# To avoid priors being over weighted, they are scaled by $\frac{K_b}{K}$, where $K_b$ is the length of a batch and $K$ is the total number of data points in a segments.
# :::

# %% [markdown]
# ## Definition: `OptimizerStatus`
#
# Optimizers may involve several sub optimization problems, most notably optimization of the parameters and the latents. Storing their state as a bit flag allows to represent any convergence state; each bit corresponds to one subproblem, and is switched to 1 when that subproblem has converged. When all bits are 1, the optimizer has converged fully.
#
# The special value -1 is used to indicate a failure which causes termination of the optimization.
#
# | Status flag | Name | Meaning |
# |------------:|------|---------|
# |  00         | `NotConverged` | Nothing has converged |
# |  01         | `ParamsConverged` | Parameter optimization has converged |
# |  10         | `LatentsConverged` | Latents optimization has converged |
# |  11         | `Converged` | Fully converged |
# | -1 = …1111  | `Failed` | Failed |
#
# Updating the convergence status is done by *bitwise OR* operations. For example, to indicate that the optimization of parameters has converged, one does
#
#     status |= OptimizerStatus.ParamsConverged
#
# Repeating a bitwise or with the same argument does not further change the value of `status`.
# Furthermore, the `Failed` status cannot be changed by bitwise OR operations. Thus it is always safe to update the status as above, independent of its current value.
#
# To check for convergence, use the *bitwise AND*. For example, the following will skip an optimization step if `status` is `Failed` or `Converged`:
#
#     if not (status & OptimizerStatus.Converged) is OptimizerStatus.Converged:
#         # do optimize step
#
# For single-bit values, the identity comparison can be dropped:
#
#     if not (status & OptimizerStatus.ParamsConverged):
#         # do optimize step
#
# The `status` variable is checked by `step()` to skip updates for already converged sub-problems. Thus it can be used as a signal to terminate a fit. Since it is a public attribute, it's value can also be checked to determine whether to call `step()` at all.

# %%
class OptimizerStatus(Flag):
    NotConverged     = 0
    ParamsConverged  = 1
    LatentsConverged = 2
    Converged        = 3  # Sum of all partial convergence statuses
    # There can only be one failed state: usage relies on the bit
    # representation of -1 being all 1's, and thus masking any OR-ed value
    Failed           = -1


# %% [markdown]
# ### Example

# %% tags=["hide-input"]
if __name__ == "__main__":
    # Set the status
    status = OptimizerStatus.NotConverged
    status |= OptimizerStatus.ParamsConverged
    assert status is OptimizerStatus.ParamsConverged
    assert (status | OptimizerStatus.ParamsConverged) is OptimizerStatus.ParamsConverged
    status |= OptimizerStatus.LatentsConverged
    assert status is OptimizerStatus.Converged
    del status


# %% [markdown]
# ## `Optimizer`

# %% [markdown]
# ### Notation conventions
#
# - Uppercase $\Theta$ / `Θ`: set of parameters
# - Lowercase $\theta$ / `θ`: parameter, i.e. $\theta \in \Theta$
# - Lowercase $\eta$ / `η`: latent variable(s)
# - $K$: Length in time bins
# - $T$: Length in continuous time
#
# ### Important optimizer attributes
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

# %% [markdown]
# ### `Optimizer` definition

# %%
class Optimizer(BaseModel, abc.ABC):

    model              : Model
    rng                : Optional[RNGenerator]=None
    data_segments      : Union[SegmentSampler]
    observed_hists     : Dict[str,History]
    latent_hists       : Dict[str,History]
    Θ                  : OptimParams   # Set with `init_params` argument
    fit_hyperparams    : ParameterSet
    update_hyperparams : Optional[Callable[[Optimizer],dict]]
    # Why can't I specify logp types as 'AccumulatedObjectiveFunction' ? When I do, they don’t
    # deserialize, and when I force them to, I get "logp_default is not a valid dict" errors.
    # As if Pydantic thought the specified type was dict ???
    logp_default       : Optional[Callable[[Integral], FloatX]]=None


    # Managed internally
    stepi               : int=0
    orig_fit_hyperparams: ParameterSet=None
    status              : OptimizerStatus=OptimizerStatus.NotConverged
        # `status` is used by subclasses
    outcome             : tuple=()

    class Config:
        arbitrary_types_allowed=True  # TODO: Remove
        allow_population_by_field_name=True  # When Θ gets exported, accept it as parameter
        fields={'Θ': {'alias': 'init_params'},
                'logp_default': {'alias': 'logp'},
                'logp_default_nodyn': {'alias': 'logp_nodyn'}}
        json_encoders = json_encoders

    """
    Base class for optimizers.

    Parameters
    ----------
    model: sinn.Model
    rng: Numpy Generator; i.e. RNG created with np.random.default_rng() or equivalent.
    data_segments:
        At the beginning of each pass, this iterator is moved one step ahead to draw
        a new data segment on which to compute gradients.
        Typically this would be an infinite generator, and each value a xarray.Dataset,
        with variable names (DataArrays) matching the names of observed histories.
        Any Mapping will work though, so values can also be a plain dictionary of
        {history name: ndarray} pairs.

        .. remark:: For convenience in test code, `data_segments` accepts
           arbitrary iterables. However these are not serializable, so when possible
           use `SegmentSampler` objects.
        observed_hists: List of observed histories.
        Observed histories are expected to be filled by data rather than integrated.
    latent_hists: List of latent histories.
        Latent histories have a random dependency but cannot be directly inferred
        from observations.
        Although they can be specified as both History instances or string names,
        strings are preferred, since they also work with histories in submodels.
        The following applies to both `observed_hists` and `latent_hists`:
           If there is only one history, it need not be wrapped with a list.
           Histories can be specified as strings, in which case the corresponding
           attribute is retrieved from `model`.
    fit_hyperparams: ParameterSet
        Must minimally define:
            'params':  Hyperparameters for the parameter optimizer
            'latents': Hyperparameters for the latents optimizer
            'Tθb': Parameter batch length, in time units
            'Tθr': Parameter relaxation time, in time units
            'Nθb': Number of batches per pass for the parameter fit
            'Tηb': Latents batch length, in time units
            'Tηr': Latents relaxation time, in time units
            'Nηb': Number of batches per pass for the latents fit
        The values of `params` and `latents` will be converted to shared
        variables when we call `compile_optimization_functions`.
    update_hyperparams: Callable, (SGDOptimizer) -> dict (Optional)
        This callback hook is to allow changing the hyperparams during a fit.
        It is called at the beginning of each pass.
        The optimizer object is passed as first argument; the function can
        access current estimates of the latents with the `.latent_hists`
        attribute, and hyperparameters with `.fit_hyperparams` and `.orig_fit_hyperparams`.
        (Recall that `.fit_hyperparams` contains shared variables.)
        The return value should be a dictionary of updates to apply to `fit_hyperparams`.
        These updated values should not be shared variables (they are applied
        with `.set_value()`).
    """

    # @classmethod
    # def _get_nested_hist(cls, histnm, model):
    #     if '.' in histnm:
    #         submodelnm, histnm = histnm.split('.', 1)
    #         return cls._get_nested_hist(histnm, getattr(model, submodelnm))
    #     else:
    #         return getattr(model, histnm)

    @initializer('observed_hists', 'latent_hists', always=True)
    def retrieve_model_hists(cls, hists, model):
        """
        Wrap bare histories with a list, and ensure listed histories are part of the model.
        Histories specified as strings are replaced by the actual history.
        """
        if isinstance(hists, (History,str)):
            hists = [hists]
        hist_dict = {}
        for h in hists:
            if isinstance(h, str):
                hist_dict[h] = getattr(model, h)
            else:
                # NB: Using hist name doesn't allow specifying hists in submodels
                assert isinstance(h, History)
                hists_dict[h.name] = h
        if not all(h in model.history_set for h in hist_dict.values()):
            raise AssertionError("The following histories are part of the model "
                                 "but not of its history_set: \n"
                                 f"{[h.name for h in hists if h not in model.history_set]}\n"
                                 "This should not happen.")
        return hist_dict

    ## Deserialize functions passed as string
    @validator('update_hyperparams', pre=True)
    def parse_callback_from_str(cls, func):
        if isinstance(func, str):
            func = mtb.serialize.deserialize_function(func)
        return func

    # TODO?: Any way to do this with two single-param validator/initializer ?
    #       It would avoid the need for values.get(...) and `if __ is None`
    #       boilerplate.
    @root_validator
    def copy_hyperparams(cls, values):
        """
        Copy hyperparams so they can change during optimization.
        The hyperparams (but not orig_hyperparams) will be made into shared
        variables when we compile the function.
        """
        hyperθ, orighyperθ = (values.get(nm, None) for nm in
            ('fit_hyperparams', 'orig_fit_hyperparams'))
        if hyperθ is None:
            return
        if orighyperθ is None:
            orighyperθ = hyperθ
        if isinstance(hyperθ, ParameterSet):
            hyperθ = hyperθ.as_dict()
        hyperθ = copy.deepcopy(hyperθ)
        values.update(fit_hyperparams=hyperθ,
                      orig_fit_hyperparams=orighyperθ)
        return values

    def copy(self, *args, deep=False, **kwargs):
        """
        Shallow copying an Optimizer is not supported: in the current
        implementation, both copies would share the same set of parameters.
        This method is intended only to allow validation by Pydantic, and
        simply returns `self`.

        .. Important: Does not actually copy the object. Appropriate only for
           factory functions or other situations where the original object would
           be discarded.

        Support for the ``deep=True`` may be added in the future.
        """
        if deep:
            raise NotImplementedError("Deep copying of {type(self).__name__} "
                                      "is not currently supported.")
        return self

    def dict(self, *args, **kwargs):
        """
        When exporting as a dict, replace lists of histories by lists of
        names. This avoids duplicated data, since the histories are already
        part of 'model', and parsing is already taken care of by the validator.

        Also, `fit_hyperparams` are often modified during a fit, e.g. to
        scale the learning rate to the stationary variance. In order to
        be able to recreate the same Optimizer from a dict export, we return
        the hyperparams that were originally passed as arguments
        """
        Store = namedtuple('Store', ['observed_hists', 'latent_hists'])
        store = Store(self.observed_hists, self.latent_hists,
                      self.logp_params, self.logp_latents,
                      self.logp_latents_nodyn)
        self.observed_hists = list(self.observed_hists.keys())
        self.latent_hists = list(self.latent_hists.keys())

        d = super().dict(*args, **kwargs)
        # Use original values of 'fit_hyperparams'
        d['fit_hyperparams'] = self.orig_fit_hyperparams
        del d['orig_fit_hyperparams']

        self.observed_hists = store.observed_hists
        self.latent_hists = store.latent_hists
        return d

    ## Abstract methods ##
    @abc.abstractmethod
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
        # TODO: This could be made optimizer-agnostic, and implemented her
        raise NotImplementError

    @abc.abstractmethod
    def step(self):
        """
        Update the parameters by performing one optimization step.

        **Side-effects**

        - Update parameters
        - Update latent histories
        """
        raise NotImplementedError

# %% [markdown]
# ## `OptimParams`
#
# `OptimParams` is a subclass of `sinn.ModelParams`. Whereas an instance of `ModelParams` (there is one for each model) stores parameters in _model space_, an instance of `OptimParams` stores parameters in _optimization space_ (see [Priors](priors)). For both technical and semantic reasons it is best to use difference classes for both spaces.

# %% tags=["hide-input"]
class OptimParams(ModelParams):
    """
    Differences with ModelParams:
    - A `ModelParams` subclass will refuse any parameter it does not recognize.
      `OptimParams` assumes any keyword argument is a valid parameter.
    - All parameters are Shared variables.
      (Models may define their `ModelParams` such that they accept
      other types, such as PyMC3 random variables.)
    - Nested parameter sets are flattened, using dotted names to indicate
      their hierarchy.
    """
    class Config:
        extra = 'allow'
    # DEVNOTE: It seems that the order of non-field attributes (those allowed
    # by extra = 'allow') is undefined. Even when attributes are added in the
    # same order, they may be returned in different orders when iterating.
    # This is why we define __iter__ and dict() below.

    # Since the set of fields is unknown, we need to be a bit more clever to
    # ensure all attributes are shared variables:
    # - Convert any keyword arg passed during instantiation to shared
    # - Convert any direct assignment to shared
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for θnm, θ in self:
            if json_like(θ, 'Array'):
                θ = Array.validate(θ)
            if isinstance(θ, dict):
                nestedΘ = OptimParams(**θ)
                for nestednm, nestedθ in nestedΘ:
                    setattr(self, f"{θnm}.{nestedθ}", nestedθ)
                delattr(self, θnm)
            elif not shim.isshared(θ):
                # NB: Repeated from __setattr__ to ensure `θnm` is used for the name
                setattr(self, θnm, shim.shared(θ, name=θnm))
    def __setattr__(self, attr, val):
        if json_like(val, 'Array'):
            val = Array.validate(val)
        if not shim.isshared(val):
            val = shim.shared(val, name=attr)
        super().__setattr__(attr, val)

    # Ensure values are always returned in a consistent order
    # (Even when values are always added in the same order,
    # the iteration order is not conserved)
    # (Verified: We really do need to force ordering with both __iter__ and dict)
    def __iter__(self):
        values = {k: v for k,v in super().__iter__()}
        for k in sorted(values):
            yield k, values[k]
    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        return {k: d[k] for k in sorted(d)}

    # Remove checks in set_values, since they are meaningless without __fields__
    def set_values(self, values: ModelParams):
        # Ideally, we would just set flags and use the parent's method,
        # but we also need to skip vars which aren't optimized
        #return super().set_values(
        #    values, must_set_all_params=False, must_not_set_other_params=False)
        if isinstance(values, dict):
            values_dict = values
        elif isinstance(values, SimpleNamespace):
            values_dict = values.__dict__
        elif isinstance(values, type(self)):
            values_dict = {k: v for k,v in values}
        else:
            # `self` is always a subclass, and we want `values` to be instance of that subclass
            raise TypeError(f"{type(self)}.set_values: `values` must be an "
                            f"instance of {type(self)}, but is rather of type "
                            f"{type(values)}.")
        for k, v in values_dict.items():
            self_v = getattr(self, k, None)  # <- This is the bit preventing
            if self_v is None:               #    preventing use of super()
                # This model parameter is not optimized (probably constant)
                continue
            if shim.isshared(self_v):
                self_v.set_value(v)
            else:
                setattr(self, k, v)

# %% [markdown]
# ## Utilities

# %%
def clip_gradients(gradients: List["SymbolicExpression"], clip: float):
    """
    Clip gradients using an L∞ norm, so the direction is conserved.
    Specifically, each gradient is independently divided by `clip`; the largest
    of these ratios, if it exceeds 1, is used to rescale the whole gradient.
    """
    # TODO: Allow gradients to be modified in place ?
    rescale = shim.max([1] + [shim.max(abs(g / clip)) for g in gradients])
        # Adding one to the list is equivalent to only clipping if `clip` is exceeded
    rescale.name = "rescale"
    # rescale = shim.print(rescale) # DEBUG
    gradients = [g/rescale for g in gradients]
    return gradients


# %% [markdown] tags=["remove-cell"]
# ## Wrap-up

# %% tags=["remove-cell"]
Optimizer.update_forward_refs()
