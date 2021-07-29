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
# # Recorders
#
# A `Recorder` follows a similar pattern to e.g. the NEST [`multimeter`](https://nest-simulator.readthedocs.io/en/nest-2.20.1/guides/analog_recording_with_multimeter.html?highlight=multimeter): a callback function with a certain recording interval.¹  Arbitrarily many recorders can be attached to a model, making this approach simple and flexible. Each recorder should have a unique name, which is used to differentiate their outputs.
#
# ¹ With the important distinction viz. the NEST multimeter that the `Recorder` is not performance-critical code, and the implementation almost trivial.
#
# ## Use case
#
# A `Recorder` object is periodically asked during training to store a value (which may be the loss, parameters, etc.). It has a few advantages over simply storing those values as attributes of the model:
#
# - It avoids the need to create a new model just to record different values.
# - It provides a mechanism for specifying the recording interval.
#   + Each recorder can have its own interval, so for example a scalar loss may be recorded
#     more often than the set of large arrays constituting the model parameters.
#   + Recording intervals may be logarithmic (in fact this is the default).
#     This allows for high resolution recording at the beginning of training, when changes
#     are biggest, and lower resolution when the training has almost stabilized.
# - It can be serialized as JSON. This makes it easy to write and read results to disk.
# - It provides convenience methods for accessing subsets of values (e.g. the set of values associated to the $w_1$ parameter).
# - The standard storage format makes it easier to write generic code.
#
# ## Storage format
#
# `Recorders` maintain two lists, which are included in the JSON serialization:
#
# - `steps`: List of `int`. The training steps at which values were recorded.
# - `values`: List of recorded values. Each entry is in correspondence with an entry in `steps`. If values are tuples, each element can be provided a name by setting `keys`. (This is useful e.g. when multiple parameters are saved together.)
# - `keys`: List of `str`. A name for each element of the saved values. (Only meaningful if `values` is a tuple.)
#
# :::{note}
# When defining a new recorder, the type of `value` should be specified (use `Any` if the type is unknown); see the definitions of `LogpRecorder`, `ΘRecorder` and `LatentsRecorder` below for examples. Specifying an explicit type helps ensure that recorded values are deserialized properly when loading data.
# :::
#
# ## Usage
#
# A training routine should call the `.ready(…)` method within its training loop; whenever it returns ``True``, it should then proceed to call `.record(…)`. This allows the training routine to force recording even when the recording condition isn't met – e.g. when training terminates. For example, the following pseudo-code ensures that the initial and final states are recorded::
#
# ```python
# def train(model, …):
#     …
#     # Ensure initial state is recorded
#     for recorder in model.recorders:
#         recorder.record(step, model)
#     # Train
#     while training:
#         step += 1
#         …
#         for recorder in model.recorders:
#             if recorder.ready(step):
#                 recorder.record(step, model)
#     # Ensure final state is recorded
#     for recorder in model.recorders:
#         recorder.record(step, model)
# ```

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup()

# %% tags=["hide-input"]
import math  # 10x Faster than numpy.log with scalar, plain Python types
import numpy as np
from numbers import Integral
from warnings import warn
from inspect import ismethod, signature
from functools import partial
from typing import TypeVar, Generic, Any, Optional, Union, List, Tuple, Callable
from pydantic import conint, confloat, constr, validator, root_validator
from pydantic.generics import GenericModel
import mackelab_toolbox as mtb
import mackelab_toolbox.serialize
from mackelab_toolbox.utils import LongList
from mackelab_toolbox.typing import json_like, Array
from smttask.typing import Type
from sinn.utils.pydantic import initializer

# %% tags=["hide-input"]
import sinnfull
from sinnfull.optim.base import Optimizer

# %% [markdown]
# ## Definition: `Recorder` object

# %%
__all__ = ['Recorder', 'DiagnosticRecorder',
           'LogpRecorder', 'ΘRecorder', 'LatentsRecorder']

# %%
RecorderInterval = Union[conint(strict=True, ge=1),confloat(ge=1)]

# %% [markdown] tags=["remove-cell"]
# TODO: A read-only Recorder, with reduced attributes

# %%

class RecorderList(LongList):
    """
    A specialized list which allows to index both step and value dimensions
    from recorded values. For non-scalar recorded values, we often want a
    particular component at all or many recorded steps.
    
    For example, suppose a recorder ``rec`` has a key ``'points`` with the
    following recorded values:
    ``[[0, 1], [0.866, 0.5], [1, 0], [0.866, -0.5]]``
    (Corresponding to tuples :math:`\sin(x),  \cos(x)` for x = 0, π/3, π/2, 2π/3.)
    We can retrieve the full list with
    
        >>> rec.points
        [[0, 1], [0.866, 0.5], [1, 0], [0.866, -0.5]]
        
    If ``rec.points`` were a normal list, it would only be indexable along
    the first dimension. To retrieve only the first (sin) component, one would
    either have to filter the result, or first construct a NumPy array:
    ``np.array(rec.points)[:,0]``. The latter approach is more convenient, but
    entails a potential expensive array construction.
    
    `RecorderList` solves this problem by providing a NumPy like interface
    which splits the first index and applies it separately. All of the
    following indexing formats are possible:
    
        >>> rec.points[1:3]
        [[0.866, 0.5], [1, 0]]
        >>> rec.points[:, 0]
        [0, 0.866, 1, 0.866]
        >>> rec.points[:, 0:1]
        [[0], [0.866], [1], [0.866]]
        >>> rec.points[3, 1]
        -0.5
    
    The implementation is essentially essentially the same as filtering the
    returned list.
    """
    def __getitem__(self, key):
        if isinstance(key, tuple):
            stepkey, *valuekey = key
            if len(valuekey) == 1:
                valuekey = valuekey[0]
            values = super().__getitem__(stepkey)
            if isinstance(stepkey, slice):
                return [v[valuekey] for v in values]
            else:
                return values[valuekey]
        else:
            return super().__getitem__(key)

# %%
ValueT = TypeVar('ValueT')
class Recorder(GenericModel, Generic[ValueT]):
    __slots__ = ('orig_callback',)
    name    : str
    keys    : Optional[Tuple[str,...]]
    interval: RecorderInterval = 1.1
    interval_scaling: constr(regex=r'^(linear|log)$')='log'
    also_record: list=[]   # Wish: `List[Recorder]`. Problem: it causes recursion error
    callback: Callable[[Optimizer], Any]=None
    # Internal
    steps   : List[int]=[]
    values  : List[ValueT]=[]
    class_type: Type=None

    class Config:
        allow_population_by_field_name = True  # Allow both 'callback' and 'record'
        validate_assignment = True
        json_encoders = sinnfull.json_encoders

    """
    Parameters
    ----------
    name: Unique label used to identify this recorder (e.g. 'log L').
        May be used as a default for figure labels.
    record: The function for which the return value is recorded.
        Return value is arbitrary; if it is a container such as a tuple,
        `key` can be used to assign names to its elements.
    keys: Optional attribute. If provided, should have the same length as
        the values returned by `record`.
    interval: Number of passes between each recording.
    interval_scaling: str, one of 'linear', 'log'
        'linear': Record every `interval` number of steps.
        'log': Record every time the number of steps is increased by a factor `interval`.
               Effectively this means that recording intervals are linear
               on a logarithmic scale.
               The default of 1.1 yields 76 stops on the first 5000 steps.

    steps: The list of steps at which we recorded. (Internal variable)
    values: The list of recorded values. (Internal variable)
    """

    # @validator('callback', pre=True)
    @classmethod
    def parse_callback_from_str(cls, func):
        if isinstance(func, str):
            # Callbacks may have been defined within the class, with @staticmethod
            # Replace @staticmethod by a noop when deserializing
            def noopdecorator(f):
                return f
            func = mtb.serialize.deserialize_function(
                func, globals={}, locals={'staticmethod': noopdecorator})
        return func

    @validator('callback')
    def no_methods_from_other_classes(cls, func):
        if ismethod(func) and not func.__func__ in cls.__dict__.values():
            raise ValueError("Only methods attached to the same recorder are "
                             "allowed, because only those can be deserialized. "
                             "Use a plain function instead.")
        return func

    @validator('steps', 'values', pre=True)
    def unpack_array(cls, v):
        "It is plausible that in some cases, lists would be saved as arrays."
        if json_like(v, 'Array'):
            v = Array.validate(v)
        return v

    @validator('class_type', pre=True, always=True)
    def set_default_class_type(cls, class_type):
        if class_type is None:
            class_type = cls
        return class_type

    @root_validator
    def valid_interval(cls, values):
        interval, interval_scaling = (values.get(x, None) for x in
                                      ('interval', 'interval_scaling'))
        if interval_scaling == 'linear' and not isinstance(interval, int):
            warn("Non-integer intervals only supported for logarithmic intervals. "
                 "Casting to integer.")
            values['interval'] = int(interval)
        elif interval_scaling == 'log' and interval == 1:
            warn("Logarithmic intervals of 1 are ill-defined. Switching "
                 "to linear intervals.")
            values['interval_scaling'] = 'linear'
        return values

    def __new__(cls, *args, **kwargs):
        # The code here redirects to the correct Recorder subclass, if possible
        if kwargs.get('class_type', None) is not None:
            class_type = Type.validate(kwargs['class_type'])
            assert isinstance(class_type, type)
            if (issubclass(class_type, cls)    # This avoids infinite recursion
                  and class_type is not cls):  # This as well
                return class_type.__new__(class_type, *args, **kwargs)
        __new__ = super().__new__
        if __new__ is object.__new__:
            # Object.__new__ fails if given any argument except `cls`.
            return __new__(cls)
        else:
            # Subclasses of Recorder end up here
            # At the very least, Recorder.__new__ is above in the MRO,
            # so we pass arguments.
            return __new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        # I haven't figured out a way to bind 'self' within a validator
        callback = kwargs.get('callback', None)
        # If we use the recorder's default_callback, we serialize it to str and
        # deserialize it back. We do this for two reasons:
        # 1) Consistency between in-memory and deserialized tasks
        # 2) If we attach the method to the recorder, we get infinite recursion:
        #    The method repr includes the repr of the Pydantic model instance,
        #    the Pydantic model repr include the repr of its attributes
        if callback is None:
            callback = getattr(self, 'default_callback', None)
            if ismethod(callback):
                callback = callback.__func__
            if callback is not None:  # Allow to be unset; e.g. for read-only Recorder
                callback = mtb.serialize.serialize_function(callback)
        # In order for the deserialized callback to still be serializable,
        # we keep the original argument (otherwise each deserializaton would add 'partial')
        object.__setattr__(self, 'orig_callback', callback)
        if isinstance(callback, str):
            callback = self.parse_callback_from_str(callback)
            if next(iter(signature(callback).parameters)) == 'self':
                # Function is a method: Bind its first argument
                # Thanks to `no_methods_from_other_classes`, we can be sure that
                # `self` refers to this instance.
                callback = partial(callback, self)
        kwargs['callback'] = callback
        super().__init__(*args, **kwargs)

    def copy(self, *args, **kwargs):
        m = super().copy(*args, **kwargs)
        object.__setattr__(m, 'orig_callback', self.orig_callback)
        return m

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        if self.orig_callback:
            d['callback'] = self.orig_callback
        return d

    def __len__(self):
        return len(self.steps)

    @property
    def last_step(self):  # Mostly for consistency with 'last'
        return self.steps[-1]
    @property
    def last(self):
        return {k: v for k,v in zip(self.keys, self.values[-1])}

    def __getitem__(self, key):
        """
        If `key` is a string:
            Return the array of values matching the given key. When all recorded values
            are scalar, this is equivalent to

            >>> np.array(self.values)[:,index_of(key)]

        If `key` is an integer:
            Return a dictionary of {key: value} pairs at that step index.
            Equivalent to

            >>> idx = self.steps.index(key)
            >>> {k:v for k,v in zip(self.keys, self.values[idx])}
            
            Exception: If the recorder does not define keys, then
            ``self.values[idx]`` is simply returned.

        If `key` is a list:
            Triggers Numpy-like fancy-indexing: a list of arrays is returned,
            one for each key. Defined as

            >>> [self[subkey] for subkey in key]

            .. note:: Always returns a list, even if there is only one key.

        If `key` is a tuple: (experimental)
            Key may be composed of a combination of integers (steps)
            or strings (values keys). Values matching one step and one value
            key are returned.
            Special cases:
               - If no value key is given, all values are returned for the
                 specified steps
               - If no steps are given, all values are returned for all the
                 specified values.
               - If there is only one step, the list for the step axis is removed
               - If there is only key value, the list for the value axis is removed
            TODO: Finish documenting, incl. format of returned val.
            TODO?: Indexing with a tuple of exactly length two, each a list –
                  one for step indices, the other for values.
                  Always returns list of lists of values
        """
        if isinstance(key, list):
            return [self[subkey] for subkey in key]
        elif isinstance(key, tuple):
            step_idcs = [self.steps.index(int(k)) for k in key
                         if isinstance(k, Integral)]
                # Casting int allows for numpy integers
            val_keys = [k for k in key if isinstance(k, str)]
            retval = None
            if len(step_idcs) + len(val_keys) != len(key):
                raise IndexError(
                    "When indexing a Recorder with a tuple, each element must "
                    f"be either an int or a string. Received: {key}.")
            if len(val_keys) == 0:
                # No value specified: return all of them
                retval = [self.values[i] for i in step_idcs]
            elif len(step_idcs) == 0:
                # No step specified: return all of them
                retval = [self[vk] for vk in val_keys]
            else:
                # Return values matching both step and value keys
                values = self.values
                val_idcs = [self.keys.index(vk) for vk in val_keys]
                retval = [[values[si][vi] for vi in val_idcs]
                          for si in step_idcs]
            # Remove list dimensions if they contain only one element
            if len(step_idcs) == 1 and len(val_keys) == 1:
                retval = retval[0][0]
            elif len(step_idcs) == 1:
                retval = retval[0]
            elif len(val_keys) == 1:
                retval = [v[0] for v in retval]
            # Return result
            return retval
        else:
            if isinstance(key, Integral):
                idx = self.steps.index(int(key))
                # NB: Casting int allows for numpy integers
                if self.keys:
                    return {k:v for k,v in zip(self.keys, self.values[idx])}
                else:
                    return self.values[idx]
            else:
                if self.keys is None and key in ("", self.name):
                    # Special case: recorders without keys use their name as a key proxy
                    # This allows a more standardized interface
                    if isinstance(self.values, LongList):
                        return self.values
                    else:
                        return LongList(self.values)
                try:
                    i = self.keys.index(key)
                except ValueError:
                    raise KeyError(f"Recorder does not contain the key '{key}'. "
                                   f"Recognized keys: {self.keys}")
                else:
                    return RecorderList(v[i] for v in self.values)

    def __getattr__(self, attr):
        """Single traces can also be retrieved by attribute instead of by indexing."""
        if self.keys and attr in self.keys:
            return self[attr]
        raise AttributeError(f"Recorder does not recognize the attribute '{attr}'.")

    def __hash__(self):
        return hash(self.json())

    # Specialize representations so they shorten the often long `values`
    # Affects __str__, __repr__ and __pretty__
    def __repr_args__(self):
        for k, v in self:
            if k in ['steps', 'values']:
                yield k, LongList(v)
            else:
                yield k, v

    def trace(self, *keys, squeeze=True):
        """
        Return a tuple suitable for plotting, composed of (steps, (trace1, trace2)).
        If there is only one key, and `squeeze` is True, the result is squeezed;
        i.e. one gets (steps, trace).
        """
        if len(keys) == 0:
            keys = self.keys or [self.name]
        if len(keys) == 1 and squeeze:
            return self.steps, self[keys[0]]
        else:
            return self.steps, self[list(keys)]
            
    def ready(self, step):
        """
        This function is called to determine whether to record.
        .. note:: A better function name might be "record_condition_met", but
           I find that too verbose. If I think of a better name I may change it.
        """
        if self.interval_scaling == 'log':
            return (step < 2 or (int(math.log(step,self.interval))
                                 - int(math.log(step-1,self.interval))))
        else:
            return (step % self.interval == 0)

    def record(self, step, optimizer):
        # Assumes steps are monotonically increasing
        if len(self.steps) == 0 or self.steps[-1] != step:
            self.steps.append(step)
            self.values.append(self.callback(optimizer))
            # By including this loop inside the conditional, we prevent cycles between recorders
            for recorder in self.also_record:
                recorder.record(step, optimizer)

    def clear(self):
        self.steps = []
        self.values = []

    def drop_step(self, step):
        """Remove the specified step from the lists of steps and values."""
        try:
            stepi = self.steps.index(step)
        except ValueError:
            logger.debug(f"{self.name}.drop_step: Step {step} was not "
                         "recorded or already removed.")
        else:
            self.values.pop(stepi)
            self.steps.pop(stepi)
            assert len(self.values) == len(self.steps)

    def get_nearest_step(self, step: int, return_index: bool=False):
        """
        Return the value of the recorded step nearest to `step`.
        If there are two nearest steps, the earliest one is returned.

        .. todo:: Should we add an argument to specify a metric, e.g.
           log scaling of distance ?

        Parameters
        ----------
        step: The step we want to be nearest to.
        return_index:
            Whether to also return the index corresponding to this step in the
            recorder's list.
        """
        step_idx = np.searchsorted(self.steps, step)
        if step_idx == len(self):
            # `step` is larger than the largest available step
            step_idx -= 1
        near_step = self.steps[step_idx]
        # Searchsorted returns the index such that near_step >= step.
        # We check if there is a even closer step which is < step
        prev_step = (-np.inf if step_idx == 0
                     else self.steps[step_idx - 1])
        # TODO: Line below is where we would add different distance measure
        if abs(prev_step-step) <= abs(near_step-step):
            near_step = prev_step
            step_idx -= 1
        if return_index:
            return near_step, step_idx
        else:
            return near_step

    def decimate(self, target_steps: int=200, always_copy: bool=False):
        """
        Return a new `Recorder` where the number of steps is no more than
        twice `target_steps`.
        The first and last recorded values are never removed.

        Parameters
        ----------
        target_steps:
            The number of steps we want. Actual number may be up to twice `target_steps`.
        always_copy:
            If True, a copy of the recorder is always returned.
            If False, a copy is only made if decimation occurs.
            Default is False.
        """
        if len(self.steps) <= 2*target_steps:
            return self
        else:
            # Take care to keep first and last step
            steps, last_step = self.steps[:-1], self.steps[-1]
            values, last_value = self.values[:-1], self.values[-1]
            Δ = int(target_steps // len(self.steps))
            steps = steps[::Δ] + [last_step]
            values = values[::Δ] + [last_value]
            return self.copy(steps=steps, values=values)

# %% [markdown]
# ## Definition: `DiagnosticRecorder` object

# %%
# FIXME: Where to specify Recorder[ValueT] ? At present can't do DiagnosticRecorder[float]
class DiagnosticRecorder(Recorder):
    # Change defaults to record on every step
    interval          : RecorderInterval = 1
    interval_scaling  : constr(regex=r'^(linear|log)$')='linear'
    record_condition  : Callable[[int, int, str], bool]=None
    # Internal
    batch_starts: List[int]=[]
    contexts    : List[str]=[]
    """
    A subclass of Recorder used for diagnosing an optimizer.
    In contrast to Recorder, defaults to recording at every step.
    Moreover, diagnostic recorders are called after every *update*,
    rather than at the end of a pass. The point within each pass at
    which a call is made is indicated by the `context` argument.

    Diagnostic recorders receive two extra arguments:

    `batch_start` [int]
        Starting index of the batch
    `context` [str]
        A string indicating where the diagnostic call was made.
        One of 'θ' | 'η_default' | 'η_rightmost' | 'η_leftmost'

    Recordings can be filtered using the `record_condition` callback.
    For example, to record only latent updates, pass
    ``record_condition=lambda stepi, k0, ctx: 'η' in ctx`` when
    creating the DiagnosticRecorder.
    """
    @initializer('record_condition')
    def set_default_record_condition(cls, rec_cond):
        return cls.default_record_condition

    @staticmethod
    def default_record_condition(stepi, k0, ctx):
        return True

    def record(self, step, optimizer, batch_start, context):
        # Assumes steps are monotonically increasing
        if (len(self.steps) == 0
            or (self.steps[-1] != step or self.contexts[-1] != context
                or self.batch_starts[-1] != batch_start)):
            self.batch_starts.append(batch_start)
            self.contexts.append(context)
            self.steps.append(step)
            self.values.append(self.callback(optimizer))
            # By including this loop inside the conditional, we prevent cycles between recorders
            for recorder in self.also_record:
                recorder.record(step, optimizer)

    def ready(self, step, batch_start, context):
        return (super().ready(step)  # Do first in case record_condition has internal state
                and self.record_condition(step, batch_start, context)
                )

    def clear(self):
        self.batch_starts = []
        self.contexts = []
        if hasattr(self.record_condition, 'clear'):
            self.record_condition.clear()
        super().clear()


# %%
Recorder.update_forward_refs()
DiagnosticRecorder.update_forward_refs()


# %% [markdown]
# ## Concrete recorders
#
# The Recorders below are the most commonly used, and serve as templates.
# Of course it is always possible to write a customized one, e.g. which
# only records a subset of the latent variables.
#
# > **NOTE** Implementations are responsible for ensuring that the value returned by their callback matches that declared as their value type. Otherwise, recorders may fail to deserialize correctly. For example, `LogpRecorder` must cast its returned value as `float`, since the value returned by `optimizer.logp()` is often a NumPy scalar.

# %%
class LogpRecorder(Recorder[float]):
    name    ='log L'
    interval: RecorderInterval=1.05

    @validator('values', pre=True)
    def temporary_workaround_for_values_stored_as_array(cls, v):
        try:
            is_array = (v[0][0] == "Array")
        except:
            pass
        else:
            if is_array:
                v = [Array.validate(logp) for logp in v]
        return v

    @staticmethod
    def default_callback(optimizer):
        return float(optimizer.logp())

# %%
class ΘRecorder(Recorder[Tuple[Array,...]]):
    """
    Instantiate either as::

        Θ_recorder = ΘRecorder(optimizer)

    or::

        Θ_recorder = ΘRecorder(keys=('θ1', 'θ2'...))

    In the first form, keys are inferred from `optimizer.model.params`;
    all model parameters are then recorded.

    To record only a subset of parameters, use the second form, the values to
    `keys` should match parameter names in `optimizer.model.params`.
    """
    # NOTE: Although one might expect it to suffice to record only inferred
    # parameters, because of the `remove_degeneracies` call, other parameters
    # may also be modified during the fit. It then becomes next to impossible,
    # short of re-running the fit, to reconstructed what those parameters were.
    # NOTE2: Since we no longer call `remove_degeneracies`, it should be safe
    # to save only changinge parameters now.

    name    ='Θ'
    interval: RecorderInterval=1.1

    def __init__(self, optimizer=None, **kwargs):
        if 'keys' not in kwargs:
            kwargs['keys'] = tuple(optimizer.Θ)
            #kwargs['keys'] = tuple(θname for θname, θ in optimizer.model.params)
        super().__init__(**kwargs)

    def default_callback(self, optimizer):
        #values = optimizer.model.params.get_values()
        values = optimizer.Θ.get_values()
        return tuple(values[θname] for θname in self.keys)

# %%
class LatentsRecorder(Recorder[List[Array['float64']]]):
    name    ='latents'
    interval: RecorderInterval=1.65
    segment_keys: List[tuple]=[]
        # TODO?: Option not to record 'segment_key' ? Would that be useful ?

    @validator('segment_keys', pre=True)
    def unpack_array(cls, v):
        "It is plausible that in some cases, key elements could be saved as arrays."
        v = [[Array.validate(w) if json_like(w, 'Array') else w for w in key]
             for key in v]
        return v

    # NOTE: The synchronization with the ΘRecorder doesn't currently work when
    # recorders are deserialized, because the optimizer is then missing.
    def __init__(self, optimizer=None, **kwargs):
        if 'keys' not in kwargs:
            # getattr allows both str and histories
            kwargs['keys'] = [hname for hname in optimizer.latent_hists]
        # Figure out if there is already a Θ_recorder attached to the optimizer,
        # and if so, ensure it is triggered every time we record latents
        if 'also_record' in kwargs:
            if optimizer is not None:
                logger.warning("LatentsRecorder ignores `optimizer` argument "
                               "if `also_record` is also passed.")
            Θ_recorder = None
        elif optimizer is None:
            if sinnfull.config.view_only:
                logger.warning("LatentsRecorder expects an `optimizer` argument "
                               "unless sinnfull.config.view_only is False. "
                               "Parameter and latent recorders will not be "
                               "synchronized.")
            Θ_recorder = None
        else:
            try:
                i = [r.name for r in optimizer.recorders].find('Θ')
            except (ValueError, AttributeError):
                Θ_recorder = None
            else:
                Θ_recorder = optimizer.recorders[i]
        if Θ_recorder and 'also_record' not in kwargs:
            kwargs['also_record'] = [Θ_recorder]
        super().__init__(**kwargs)

    # TODO: Extend ready() so that new segments are recorded independent of step

    @staticmethod
    def default_callback(optimizer):
        return tuple(h.get_data_trace(include_padding=True) for h in optimizer.latent_hists.values())

    def record(self, step, optimizer):
        # Copied from Recorder.record
        # Assumes steps are monotonically increasing
        if len(self.steps) == 0 or self.steps[-1] != step:
            self.steps.append(step)
            self.values.append(self.callback(optimizer))
            self.segment_keys.append(optimizer.current_segment_key)
            # By including this loop inside the conditional, we prevent cycles between recorders
            for recorder in self.also_record:
                recorder.record(step, optimizer)

# %% [markdown]
# ---
# ## Example

# %%
if __name__ == "__main__":
    import random

    class DummyOptimizer:
        def logp(optimizer):
            return random.gauss(0,1)
    optim = DummyOptimizer()

# %%
if __name__ == "__main__":
    recorder = LogpRecorder(interval=1.3, optimizer=optim)

    for step in range(100):
        if recorder.ready(step):
            recorder.record(step, optim)

# %% [markdown]
# Note how the logarithmic recording interval provides higher resolution for the initial steps.

# %%
if __name__ == "__main__":
    print(recorder.steps)
    print(recorder.values)

# %%
if __name__ == "__main__":
    import numpy as np

    class DummyNPOptimizer:
        def logp(optimizer):
            return np.random.normal(0,1,size=())  # Returns a NumPy scalar
    optim = DummyNPOptimizer()
    recorder_np = LogpRecorder(interval=1.3, optimizer=optim)

    for step in range(100):
        if recorder_np.ready(step):
            recorder_np.record(step, optim)

# %%
if __name__ == "__main__":
    print(recorder_np.steps)
    print(recorder_np.values)

# %%
if __name__ == "__main__":
    print(recorder.json())

# %% [markdown]
# The Recorder subclass is saved along with the data, and instantiation redirects to the that subclass, even when we use the base `Recorder` class.
# > Recorders defined in "\_\_main\_\_" are not deserializable, because their module is unknown.

# %%
if __name__ == "__main__":
    from sinnfull.optim.recorders import (  # Use imported recorders so they are deserializable
        Recorder as Lib_Recorder, LogpRecorder as Lib_LogpRecorder)

    optim = DummyOptimizer()
    recorder = Lib_LogpRecorder(interval=1.3, optimizer=optim)

    for step in range(100):
        if recorder.ready(step):
            recorder.record(step, optim)

# %%
if __name__ == "__main__":
    new_recorder = Lib_Recorder.parse_raw(recorder.json())
    assert isinstance(new_recorder, Lib_LogpRecorder)
    assert str(new_recorder.values) == str(recorder.values)
