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
# # Record store viewer

# %% [markdown]
# The `RSView` class is the primary interface for interacting with the Sumatra record store. It provides methods for obtaining a rapid overview of the record store contents – histograms of run timestamps and durations – as well as more fine-grained summaries – tabulating which initial conditions were used with which hyperparameters, plotting ensembles of fit curves, plotting latents.
#
# An `RSView` provides the `splitby` method to split it into a set of disjoint RSView. The default arguments will according to model, latent and observed histories – this is appropriate for the most typcial use case, since it rarely makes sense to compare fits for different models except in aggregate. The `splitby` creates a type `SplitKey`, which is a tuple of the field names differentiating the split views; instances of `SplitKey` are used as keys in the dictionary of split `RSViews`.

# %% [markdown]
# Fits are loaded into a `FitData` object, which provides convenience methods for retrieving the information required for producing summaries and plots – the model class object, hyperparameters ($\Lambda$), initialization key (*init_key*), training data, ground truth parameters, etc. Each fit is associated to `FitData.FitKey`, which is a tuple composed of $\Lambda$ and *init_key*.

# %% [markdown]
# Multiple fits can be organized into `FitCollection`s. These are loose collections, which in contrast to an `RSView`, need not be disjoint; they are created by specifying an arbitrary list of fits and a unique key. They are intended to be computationally cheap, caching and reusing `FitData` instances when possible. The main purpose of `FitCollection`s is for plotting ensembles of fits.
#
# In addition to their less rigid structure, the main difference between a `FitCollection` and an `RSView` is that iterating over the former produces `FitData` instances, whereas iterating over the latter produces Sumatra `Record` instances.
#
# `FitCollections` also provide a `split_by_Λ` method, to split it into disjoint sub collections sharing the same hyperparameters. In theory one could use `RSView`'s `splitby` to achieve this, but it is much simpler to implement with `FitData` rather than `Record` objects.
#
# One can obtain a `FitCollection` composed of all the fits of an `RSView` instance by accessing its `.fitcoll` property; this is a cached property which is created on first access. In this case the view's `SplitKey` is automatically associated as the `FitCollection`'s key.

# %% [markdown]
# | Object | Key type | Key description |
# |---|---|---|
# |`RSView` | `SplitKey`<br>`RootKey` | Dynamic; default is the minimal subset of (*model*, *latent vars*, *observed vars*). Special `RootKey` is used for the unsplit view |
# |`FitCollection` | `SplitKey` (if created from `RSView`) <br> `SplitKey` + `Λ` (if created with `split_by_Λ()`) <br> *user defined* (otherwise) |
# |`FitData` | `FitKey` | ($\Lambda$, *initialization key*) |
# |          | `θKey`  | (*θ name*, *θ index*) – e.g. $(W, (0,0))$. Used to distinguish plot panels. |
#
# Key types are all subclasses of `namedtuple`, so they have field names and values. They can be combined into larger keys with `join_keys()`. (**TODO**: allowing keys to be combined with the '+' operation while preserving field names.)
#
# The special `RootKey` type has no fields and therefore its only possible value is an empty tuple. We declare an instance `root_key` in the module scope, which can be used by any method.
# (**TODO**: the RootKey type has a `label` attribute equal to “all records”, but I'm not sure this is the best way to do this.)

# %% [markdown]
# > In order to behave correctly with HoloViews, key values should not be tuples. When this is the case (as for an *init key* and *θ index*), they can be cast to `StrTuple`. These sort and compares as a tuple, but otherwise behaves as a string (so that we have $\mathrm{"}(5, 9)\mathrm{"} < \mathrm{"}(5, 10)\mathrm{"} = \mathrm{"}(5,10)\mathrm{"}$).

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup('theano', view_only=True)

# %% tags=["remove-cell"]
import logging
logger = logging.getLogger(__name__)

# %%
import numpy as np
import re                 # For Utility functions - get_init_key
import abc                # For Plotting functions - ColorEvolCurve
import itertools
import functools
from collections import defaultdict
from scipy import signal  # For decimating fit curves
import pandas as pd
from tqdm.auto import tqdm
import smttask
import mackelab_toolbox as mtb
import mackelab_toolbox.typing
from mackelab_toolbox.parameters import ParameterComparison
from smttask import Task, TaskDesc
from smttask.utils import get_task_param
from mackelab_toolbox.utils import Singleton
import sinnfull
from sinnfull.parameters import ParameterSet
from sinnfull.utils import add_to, add_property_to, model_name_from_selector
from sinnfull.viz.config import pretty_names, BokehOpts
from sinnfull.viz.typing_ import StrTuple, KeyDimensions, ParamDimensions
from sinnfull.viz.utils import get_logL_quantile
from sinnfull.viz.hooks import hide_table_index

# %% tags=["remove-cell"]
from mackelab_toolbox.utils import TimeThis
TimeThis.on = False  # Turn off timing by default

# %%
# Types
from collections import namedtuple
from typing import (Optional, ClassVar, Union, Iterable, Generator, Any,
                    Sequence, List, Callable, Dict, Tuple)
from dataclasses import dataclass, field
from pydantic import BaseModel, PrivateAttr
from smttask.view import RecordView
from sinn import Model
from sinn.utils.pydantic import initializer
from sinnfull.models import Prior, ModelSpec, get_model_class
from sinnfull.data import DataAccessor
from sinnfull.optim import Recorder
from sinnfull.optim.recorders import ΘRecorder
    # Only used as read-only container for backward transformed params
    # Using a bare Recorder segfaults

# %%
import holoviews as hv
import seaborn as sns
import matplotlib as mpl

# %% tags=["remove-cell"]
if __name__ == "__main__":
    hv.extension('bokeh')
    #hv.renderer('bokeh').theme = 'dark_minimal'
    from IPython.display import display


# %% [markdown]
# ## Key type registry
#
# HoloViews uses *key dimensions* (`kdims`) to align dimensions between plot objects. Thus it is important to have consistent keys between our objects. Each dimension has a *name* and optional *label*.
#
# The *KeyTypes* defined below are *namedtuples* where the fields match some of the key dimensions. (So different *KeyTypes* are just different subsets of the key dimensions.) Each *KeyType* has a `kdims` attribute returning the corresponding *HoloViews* `Dimension`s.

# %%
class KeyTypes(dict):
    def get(self, key_name: str, key_fields: tuple) -> Type[namedtuple]:
        assert isinstance(key_name, str)
        assert isinstance(key_fields, Sequence)
        reg_key = tuple(itertools.chain(key_name, *key_fields))
        try:
            KeyType = self[reg_key]
        except KeyError:
            KeyType = namedtuple(key_name, key_fields)
            self[reg_key] = KeyType
        return KeyType


# %% [markdown] tags=["remove-cell"]
#     # WIP: Class to be mixed with namedtuple
#     class MixinKeyType:
#         def __init_subclass__(cls):
#             # FIXME: Not flexible enough for index dimensions ?
#             if not hasattr(cls, 'kdims'):
#                 cls.kdims = [key_dims.get(nm, pretty_names.get(nm))
#                              for nm in cls._fields]
#         def __add__(self, other):
#             return join_keys(self, other)
#         def __str__(self):
#             field_str = ",".join(self._fields)
#             return f"{self.__name__}[{field_str}]"
#         def __repr__(self):
#             # TODO: Make repr valid Python def ?
#             field_str = ",".join(self._fields)
#             return f"{self.__module__}.{self.__qualname__}[{field_str}]"

# %%
def make_key(name: Union[str,namedtuple],
             dims: Iterable[Union[hv.Dimension,str]]=None) -> Type[tuple]:
    """
    Difference with `KeyTypes`: takes Dimensions as arguments instead of field names.
    TODO: Merge into `KeyTypes`.
    """
    if hasattr(name, '_fields'):
        if isinstance(name, type):
            T = name
        else:
            T = type(name)
        return make_key(T.__name__, T._fields)
    elif dims is None:
        raise TypeError("`dims` can only be omitted if `name` is a namedtuple.")
    dims = [key_dims.get(str(dim)) for dim in dims]
        # Does two things:
        # 1: Works with str and Dimension args  2: Converts iterable to list
        # Having kdims always be the same type allows to do T1.kdims + T2.kdims
    KeyType = key_types.get(name, [dim.name for dim in dims])
    KeyType.kdims = dims
    return KeyType


# %% [markdown]
# ## Hyperparameter registry
# We assign to each loaded fit a unique short name (`Λ1`, `Λ2`…) identifying the set of learning parameters used in that fit. Since we want these short names to be shared across experiments, we use a single global registry for the entire project.

# %%
@dataclass
class ΛRegistry(metaclass=Singleton):
    """
    Registry of hyperparameters. Ensures each hyperparameter set is assigned
    a unique label throughout the project.
    """
    prefix    : str='Λ'
    labels    : dict=field(default_factory=lambda: {})
    param_sets: dict=field(default_factory=lambda: {})

    def add_param_set(self, param_set: ParameterSet) -> str:
        """Redundant with 'get_label', but makes a more logical API."""
        return self.get_label(param_set, create_if_missing=True)

    def get_label(self, param_set: Union[ParameterSet,str],
                  create_if_missing: bool=True) -> str:
        """
        Return a parameter set label for `param_set`. If a label has already
        been requested for that set, it is returned. Otherwise, a new label
        is created (unless `create_if_missing` is False)
        Special case: if `param_set` is a string AND it matches one of the
        existing labels, it is simply returned.
        """
        if isinstance(param_set, str) and param_set in self.param_sets:
            # EARLY EXIT: Idempotent on labels
            return param_set
        if not isinstance(param_set, dict):
            raise TypeError("`param_set` must be a ParameterSet instance.")
        Λ_digest = paramset_digest(param_set)
        try:
            Λlabel = self.labels[Λ_digest]
        except KeyError as e:
            if not create_if_missing:
                raise e
            Λlabel = f"{self.prefix}{len(self.labels)+1}"
            self.labels[Λ_digest] = Λlabel
            self.param_sets[Λlabel] = param_set
        return Λlabel

    def get_param_set(self, label: str) -> ParameterSet:
        return self.param_sets[label]


# %% [markdown]
# ## Globals

# %%
from mackelab_toolbox.parameters import digest as paramset_digest

# %% [markdown]
# Dimension registries

# %%
key_dims = KeyDimensions()
param_dims = ParamDimensions(key_dims)  # Used in FitData.θ_curves to set the vdim for each curve
hist_dims = ParamDimensions(key_dims)

# %% [markdown]
# Key registries

# %%
Λregistry = ΛRegistry()
key_types = KeyTypes()

# %% [markdown]
# Key types

# %%
# Remark: Dimension names must be unique across all keys
RootKey = make_key('RootKey', ())  # The key for an unsplit record store; equiv to 'all'
RootKey.label = "all records"
θKey = make_key('θKey', [key_dims.get('θname', label='parameter'),
                         key_dims.get('θindex', label='parameter index')])
HistKey = make_key('HistKey', [key_dims.get('hist', label='variable'),
                               key_dims.get('hist_index', label='variable component')])

# %% [markdown]
# There is only one root key; might as well declare it as a global

# %%
root_key = RootKey()

# %% [markdown]
# ### Defaults

# %% [markdown]
# These are used as defaults for the `splitby` method. The order matters: later fields which are not required to distinguish the split record stores are discarded. In practice, this means that in most cases, *'observed'* is not used – it will not appear in default plot titles, nor be a dropdown option for HoloMaps.

# %%
default_split_fields = ['optimizer.model.model_class', 'optimizer.latent_hists', 'optimizer.observed_hists']   # column name used for splitting
default_split_dims    = [key_dims.get('model'),
                         key_dims.get('latents', label='latent vars'),
                         key_dims.get('observed', label='observed vars')]
default_split_dims    = {dim.name: dim for dim in default_split_dims}


# %% [markdown]
# ## Utility functions

# %%
def join_keys(*keys: namedtuple, key_name: Optional[str]=None) -> namedtuple:
    """
    Key fields are not allowed to be duplicated.

    As a convenience, if key fields are duplicated, and all duplicated fields
    have the same key value, the duplicated fields are silently dropped.
    """
    if not key_name:
        key_name = '_'.join(type(key).__name__ for key in keys)
    if not all(hasattr(key, '_fields') for key in keys):
        raise TypeError("join_keys: Not all provided keys are namedtuples.")
    key_fields = tuple(itertools.chain.from_iterable(key._fields for key in keys))
    key_vals = tuple(itertools.chain.from_iterable(key for key in keys))

    # Remove duplicate fields if they have the same value
    dup_fields = {}  # List of indices for each duplicated field
    for field in key_fields:
        if key_fields.count(field) > 1:
            i = key_fields.index(field)
            dup_fields[field] = [i]
            while field in key_fields[i+1:]:
                i = key_fields.index(field, i+1)
                dup_fields[field].append(i)
    if dup_fields:
        new_fields = []
        new_keys = []
        for field in key_fields:
            if field in dup_fields:
                if len(set(key_vals[i] for i in dup_fields[field])) > 1:
                    raise ValueError(f"Key provides different values for field {field}: "
                                     f"{set(key_vals[i] for i in dup_fields[field])}")
                if field not in new_fields:
                    new_fields.append(field)
                    new_keys.append(key_vals[dup_fields[field][0]])
            else:
                new_fields.append(field)
                new_keys.append(key_vals[key_fields.index(field)])
        key_fields = tuple(new_fields)
        key_vals = tuple(new_keys)

    KeyType = key_types.get(key_name, key_fields)
    return KeyType(*key_vals)


# %%
def wrap_with_tuple(k):
    """
    Useful for standardizing keys, since keys of length 1 are not
    always wrapped with a tuple.
    A better solution may be to make sure keys are always created as
    tuples, but applying this function is often a quicker fix.
    """
    if isinstance(k, str) or not isinstance(k, Sequence):
        k = (k,)
    return k


# %%
def get_init_key(record: RecordView) -> Union[StrTuple, str]:
    """
    Utility function for extracting the RNG key used to select initial
    conditions for a fit. Because this key is not stored in the record
    parameters (only the resulting initial parameter values are), we need
    to use a bit of hackery to extract it from the 'reason' string.
    """
    key = re.search(r"Init params: (.*)", "\n".join(record.reason))[1]
    if key[0] == '(':
        key = StrTuple(eval(key))
    return key


# %%
def param_str(params: dict, include: Optional[List[str]]=None, exclude: Optional[List[str]]=None) -> str:
    """
    Return a string of comma separated <param name>=<param value> pairs.
    Hierarchical parameters (whose names include a period) are abbreviated
    by keeping only the portion of their name after the last period.

    Pretty names are used when available for <param name>.

    Parameters
    ---------
    include: List of parameter names. They must match keys in `params`.
        If provided, only parameters matching these names are returned.
    exclude: List of parameter names. They need not match a key in `params`
        (if they don't match, there is nothing to exclude).
        A parameter both in `include` and `exclude` is excluded.
    """
    if include is None:
        include = params.keys()
    if exclude is None:
        exclude = ()
    nms_vals = [(pname.split('.')[-1], params[pname])
                for pname in include if pname not in exclude]
    return ', '.join(f"{pretty_names[nm]}={val}" for nm,val in nms_vals)


# %%
def make_key_label(key: Union[namedtuple,str], format: str='default') -> str:
    if isinstance(key, str):
        # Early exit: just prettify with `make_slug` and exit
        return make_slug(key_field, separable_collections=(list, set), format=format)
    if not hasattr(key, '_fields'):
        raise TypeError("`key` must be a namedtuple (or at least define '_fields').")
    label_els = []
    assert len(key) == len(key._fields)  # In case 'namedtuple' was faked with a _fields attr
    for key_field, field in zip(key, key._fields):
        key_str = make_slug(key_field, separable_collections=(list, set),
                            format=format)
        label_els.append(f"{field.capitalize()}: {key_str}")
    return " – ".join(label_els)


# %%
def make_slug(obj: Union[str, list, tuple, set],
              separable_collections = (list, tuple, set),
              format: str='unicode') -> str:
    """
    Convert an value to a short, readable plain text string.
    When possible, `pretty_names` is used to replace values with nicer strings.

    .. note:: A heuristic is used to determine whether to wrap the returned string
       with formatting indicators (e.g. $…$ for TeX strings): if any of the comma
       separated values in `obj` match an entry in the `pretty_names` NameMap, then
       the resulting string is wrapped.

    Parameters
    ----------
    obj: Value, or comma-separated list of values. May include brackets
        or parentheses (i.e. may be a str repr of a list or tuple.)
        Also accepts lists of values.
        When a list (either true list or comma-separated string),
        `pretty_names` is applied to each element separately.
    separable_collections: Which collections types to split and apply
        `pretty_names` separately to each element. By default, these
        are `list`, `tuple` and `set`. One reason to change this default
        could be to prevent separation of tuples, if those should be treated
        as atomic keys.
    format: 'unicode' | 'latex' | 'default' | 'ascii'
        One of the NameMap names in `pretty_names`.

    Returns
    -------
    Comma separated string of values.
    If no values are provided (i.e. `obj` is an empty string or list),
    then the string "∅" (symbol for the empty set) is returned.
    """
    # In order to accept both sequences and strings, we convert everything to strings
    s = str(obj)
    # Determine which prettyfied form to use based on 'format'
    name_map = getattr(pretty_names, format)
    # Determine the brackets indicating each separable collection
    # (tested with list, tuple and set; may work with other collections)
    collection_brackets = ''.join(str(list([1])).replace("1", "").replace(",","")  # `str(set())` does not produce the brackets
                                  for T in separable_collections)

    if len(s.strip(collection_brackets)):
        # Split and rejoin each collection element.
        # Caveat: This will not recognize unpaired brackets; e.g. '[1, 2)' is equiv to '[1, 2]'
        varnames = [varname.strip("' ") for varname in s.strip(collection_brackets).split(',')]
        res_str = ', '.join(name_map[varname] for varname in varnames)
        # Heuristic: wrap the joined string iff at least one element was part of name map.
        if set(varnames) & set(name_map.keys()):
            res_str = name_map.wrap(res_str)
    else:
        res_str = "∅"
    return res_str


# %%
def make_slugs(key: namedtuple) -> namedtuple:
    """
    Create a new key, of same type as `key`, applying `make_slug` to each value.
    """
    if not hasattr(key, '_fields'):
        raise TypeError("`make_slugs` expects a namedtuple.")
    return type(key)(*(make_slug(v) for v in key))


# %%
def clip_yrange(curves: Iterable[hv.Curve], low_quantile, high_quantile):
    """
    Set the ylim on a set of curves such that extreme values are clipped.
    All finite values in the curves are pooled, and quantiles computed
    on the pooled values.

    Parameters
    ----------
    curves: Ensemble of curves we want to set the ylim on.
        The iterable must not be consumable.
    low_perceptile: float between 0 and 1. Values below this are clipped.
    high_perceptile: float between 0 and 1. Values above this are clipped.
    """
    if low_quantile > high_quantile:
        raise ValueError("`high_quantile` must be larger than `low_quantile`. "
                         f"Received: {low_quantile} (low) – {high_quantile} (high)")
    if not ((0 <= low_quantile <= 1) or (0 <= high_quantile <= 1)):
        raise ValueError("`high_quantile` and `low_quantile` must be between 0 and 1. "
                         f"Received: {low_quantile} (low) – {high_quantile} (high)")
    data = sorted(itertools.chain.from_iterable(curve.data[:,1][np.isfinite(curve.data[:,1])]
                                                for curve in curves))
    L = len(data)
    ylim = (data[int(low_quantile*L)], data[min(int(high_quantile*L),L-1)])
    for curve in curves:
        curve.opts(ylim=ylim)


# %% [markdown]
# ## Colouring evolution curves
# The function factories below implement different algorithms for emphasizing certain curves from an ensemble. They all create a function taking as single argument the `HoloMap` to colour:
#
#     fit_curves = <function returning a HoloMap of fit curves>
#     coloring_function = ColorEvolCurves(<arguments>)
#     coloring_function(fit_curves)
#
# - Each frame in `fit_curves` is treated as a different ensemble.
# - Because the returned functions all have the same signature, they can be used
#   as arguments in plotting functions.
#   (Without the receiving function having to know how they are parameterized.)
#
# There are currently two implemented colouring functions:
# - `ColorEvolCurvesByMaxL` - Automatic method; usually the one to use.
#   Emphasizes the curve(s) with the highest likelihood, using some heuristics
#   to discard fluctuating fits with sporadically high likelihood.
# - `ColorEvolcurvesByInitKey` - A manual curve, for specifying exactly which
#   curve(s) to emphasize. The init keys can be obtained from the tool tile
#   shown when hovering over a figure.

# %%
class ColorEvolCurves(abc.ABC):
    # Colouring functions normally define two methods:
    # __init__ : to parse their own parameters. Signature is arbitrary.
    # __call__ : to colour the HoloMap of fit curves. Signature is fixed.

    @abc.abstractmethod
    def __call__(self, frames: hv.HoloMap) -> hv.HoloMap:
        raise NotImplementedError


# %%
class ColorEvolCurvesByMaxL(ColorEvolCurves):
    """
    Colour evols with the highest log L at given quantile with an accent colour.
    The rule is applied as follows:
        For each logL curve, all logL values are pooled and the `quantile`
        value determined (such that `quantile`=1 would give the maximum
        log L value, and `quantile`=0 would give the minimum).
        The determined value becomes the 'log L' value for that fit.
        Fit curves in `frames` are then coloured based on the difference
        between their log L value and the maximum value over all fits.

    The use of the `quantile` argument is intended to address the case where
    a fit may spike to highest likelihood only to fall back to a lower
    value, sometimes catastrophically so. This happens more often when fitting
    latents, where log L curves are more erratic.

    Parameters
    ----------

    logL_frames: HoloMap instance. ``logL_frames.kdims`` must be a subset of
        ``frames.kdims``, so that curves can be matched to their corresponding
        log likelihood.
    quantile: The logL value at this quantile assigned as a fit's 'log L'
        value (see above).
    window: Number of steps from the end to consider.
        By default, all steps are considered when determining the log L value;
        early steps can be discarded by setting this to a positive value. For
        example, if `window`=1000, only log L values in the **last** 1000 steps
        are considered.
        This can help discard initial fluctuations to focus on stabilized
        log L values, which are usually more relevant.
    """
    def __init__(self, logL_frames: hv.HoloMap,
                 quantile: float=.95, window: Optional[int]=None):
        # TODO: DRY with get_logL_quantile
        self.logL_frames = logL_frames
        self.quantile = quantile
        if quantile < 0.7 and window is None:
            logger.warning("When using a small value of `quantile` it is "
                           "recommended to also set a value for `window` in "
                           "order to discard the initial transients.")
        if window is None:
            self.window = np.inf
        else:
            self.window = window

    def __call__(self, frames: hv.HoloMap) -> hv.HoloMap:
        """
        frames: HoloMap instance. Each frame is one ensemble of curves to colour.
            If a curve in `frames` does not have a key matching one of the
            logL curves used for initialization, its logL is set to -∞.

        """
        bokeh_opts = BokehOpts()
        logL_frames = self.logL_frames
        quantile = self.quantile
        window = self.window

        # Create a dict mapping frame keys to logL_frame keys
        # => the mapping must be surjective (many frames -> one logL_frame)
        try:
            kidcs = [frames.get_dimension_index(dim.name)
                     for dim in logL_frames.kdims]
        except Exception as e:
            # Holoviews raises a plain exception, so our only choice is to inspect the message…
            if "Dimension" in str(e) and "not found in" in str(e):
                raise ValueError("The kdims of the reference HoloMap must be a "
                                 "subset those of `frames`.\n"
                                 f"Reference (self.logL_frames.kdims): {logL_frames.kdims}\n"
                                 f"frames.kdims: {frames.kdims}.")
            else:
                raise e
        test_key = next(iter(frames.keys()))
        if isinstance(test_key, str) or not isinstance(test_key, Sequence):
            # Special case: there is only a single key, and it is not wrapped
            # in a tuple. Indexing it makes no sense.
            surj_map = {key: key for key in frames.keys()}
        elif len(kidcs) == 1:
            # Special case #2: logL keys have only one component, and therefore
            # will not be wraped in a tuple.
            i = kidcs[0]
            surj_map = {key: key[i] for key in frames.keys()}
        else:
            surj_map = {key: tuple(key[i] for i in kidcs) for key in frames.keys()}
        for frame_key, frame in frames.items():
            # Select the ensemble of curves from logL_frames matching the frame key.
            logL_key = surj_map[frame_key]
            if logL_key not in logL_frames.keys():
                raise ValueError(f"`logL_frames` does not contain an entry "
                                 f"with key {logL_key}.")
            logL_curves = logL_frames[logL_key]
            # Assign the log L quantile to each curve
            logL_vals = defaultdict(lambda:-np.inf)  # -np.inf used for curves without associated logL
            for init_key in frame.keys():
                if init_key not in logL_curves.keys():
                    # Most likely this curve was excluded from colouring
                    # => It's logL will be set to -	∞
                    continue
                    # raise ValueError(f"`logL_frames[{frame_key}]` does not "
                    #                  f"contain a curve with key {init_key}.")
                curve = logL_curves[init_key]
                logL_vals[init_key] = get_logL_quantile(
                    curve.data, quantile, window)

            # Adjust the display options for each curve in the frame based on
            # distance to max log L
            max_logL = max((-np.inf, *logL_vals.values()))
            if not np.isfinite(max_logL):
                logger.warning(f"All of the curves in `logL_frames` for frame {frame_key} "
                               "have non-finite values. Colouring is not possible.")
                continue
            for init_key, curve in frame.items():
                logL = logL_vals[init_key]
                curve.opts(bokeh_opts.dyn_accent_curve(max_logL - logL))


# %%
class ColorEvolCurvesByInitKey(ColorEvolCurves):
    """
    Colour each curve matching one of the given `init_keys` with the accent colour.

    .. Note:: In contrast to other colouring functions, no style is applied to
       ensemble curves, to allow applying this method multiple times
       (e.g. to emphasize different fits with different colours).

    Parameters
    ----------
    init_keys: Curve keys indicating which curves to emphasize.
        Any valid value for ``frames.select()`` is accepted. The limitations of
        `~hv.HoloMap.select` also apply: notably, tuples are treated as ranges.
    accent_style: (Optional) The Curve style to use for emphasis. By default,
        ``BokehOpts().accent_curve`` is used. Construct with ``hv.opts.Curve(…)``.
    """
    def __init__(self, init_keys, accent_style: hv.Options=None):
        if accent_style is None:
            self.accent_style = BokehOpts().accent_curve

    def __call__(self, frames: hv.HoloMap) -> hv.HoloMap:
        """
        frames: HoloMap instance. Each frame is one ensemble of curves to colour.
        """
        accent_style = self.accent_style
        frames.select(init_key=init_key).opts(accent_style)


# %% [markdown]
# ## `FitData` definition

# %%
class FitData(BaseModel):
    #Class attrs
    """
    Store and compute fit data required for analyses.

    .. Note:: Within analysis pipelines, one should almost always create
       `FitCollections` rather than creating `FitData` instances directly.
       Among other things, this ensures that hyperparameter labels are
       consistent among `FitData` instances.
       Directly instantiating individual `FitData` objects is mostly useful for
       debugging; it is similar to but much faster than the idiom
       `next(iter(rsview.fitcoll))`.

    Parameters
    ----------
    record: The Sumatra record associated to the fit.
    model_name: Name of the model that was fit.
    Λlabel: The short label name associated to the hyperparameter set for this
       fit. The special value "Λ?" (default) is used to indicate that this set
       is unknown. For values other than "Λ?", can be used to associate fits
       performed with the same hyperparameters.
       Appropriate `Λ` and `Λlabel` values are generated automatically by
       `FitCollection`.
    Λ: The set of hyperparameters used for this fit. This is generally a
       reduced set, containing only parameters that differ between fits.
       The default is an empty dictionary, which is appropriate only if all
       considered fits use the same parameters. It is provided manly for
       convenience when instantiating a `FitData` directly.
    """
    data_accessor_registry : ClassVar[dict] = {}
    model_registry : ClassVar[dict] = {}
    FitKey = make_key('FitKey', ['Λ', 'init_key'])
    kdims: ClassVar[List[hv.Dimension]] = [
        key_dims.get('Λ', label='hyper params'),
        key_dims.get('init_key', label='init key')]  # FIXME: Make immutable
    #Instance attrs
    record       : RecordView
    model_spec   : ModelSpec=None
    model_name   : str=None
    Λlabel       : str="Λ?"
    Λ            : dict={}
    logL_evol    : Recorder=None
    Θ_evol       : Recorder=None
    latents_evol : Recorder=None
    # Private memoization attributes  (can't use lru_cache b/c FitData is not hashable)
    _logL_curve  : hv.Curve=PrivateAttr(default=None)
    _θ_curves    : hv.HoloMap=PrivateAttr(default=None)
    _gt_η_curves : hv.HoloMap=PrivateAttr(default=None)
    _prior       : Prior=PrivateAttr(default=None)
    #latents_curve: hv.Curve
    _ground_truth_Θ: dict = PrivateAttr(default_factory=lambda:{})
    _segment_sampler: SegmentSampler = PrivateAttr(default=None)

    class Config:
        arbitrary_types_allowed=True

    @initializer('model_spec')
    def get_model_spec(cls, value, record):
        model = get_task_param(record, 'optimizer.model')
        if isinstance(model, Model):
            # Deserialized model
            raise NotImplementedError
        else:
            # Serialized model
            return get_task_param(model, 'model_selector')

    @initializer('model_name')
    def get_model_name(cls, value, model_spec):
        return model_name_from_selector(model_spec)
    # @initializer('model_name')
    # def get_model_name(cls, value, record, model_spec):
    #     model = get_task_param(record, 'optimizer.model')
    #     if isinstance(model, Model):
    #         # Deserialized model
    #         return model.name
    #     else:
    #         # Serialized model
    #         model_sel = get_task_param(model, 'model_selector')
    #         return model_name_from_selector(model_sel)
    #
    @initializer('logL_evol')
    def get_logL_evol(cls, value, record):
        output = record.get_output('log L')
        # TEMPORARY: Workaround for some losses that were saved as a list of scalar arrays
        if len(output.values) and mtb.typing.json_like(output.values[0], 'Array'):
            output.values = [float(mtb.typing.Array.validate(l)) for l in output.values]
        return output
    @initializer('Θ_evol')
    def get_optimΘ_evol(cls, value, record) -> Recorder:
        optimθrecorder = (record.get_output('Θ') or record.get_output('θ')
                          or record.get_output('params'))
        return optimθrecorder
            # Will be replaced by a modelθrecorder in __init__

    @initializer('latents_evol')
    def get_latents_evol(cls, value, record) -> Recorder:
        latents_recorder = record.get_output('latents')
        # HACK: Recorder should be able to deserialize itself
        try:
            if next(iter(mtb.utils.flatten(latents_recorder.values))) == 'Array':
                latents_recorder.values = [[mtb.typing.Array['float64'].validate(v) for v in vallist]
                                           for vallist in latents_recorder.values]
        except StopIteration:
            pass
        return latents_recorder

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.key})"
    def __repr__(self) -> str:
        return type(self).__name__ + '(' + ', '.join((
            f"key={self.key}",
            f"record={self.record.label}",
            f"Λ={'{}' if len(self.Λ) == 0 else '<…>'}",
            f"Λlabel={self.Λlabel}",
            f"logL_evol={'Recorder(…)' if isinstance(self.logL_evol, Recorder) else self.logL_evol}",
            f"Θ_evol={'Recorder(…)' if isinstance(self.Θ_evol, Recorder) else self.Θ_evol}",
            f"latents_evol={'Recorder(…)' if isinstance(self.latents_evol, Recorder) else self.latents_evol}"
            )) + ')'

    def __init__(self, *, θspace='model', **kwargs):
        super().__init__(**kwargs)
        # Parameters are recorded in optimization space, but it is preferred
        # to display them in model space.
        # So we create a new recorder with converted values
        # TODO: Can't we move this to the validator for Θ_evol ?
        assert θspace in ['model', 'optim']
        if θspace == 'model':
            optimθrecorder = self.Θ_evol
            optim_names = optimθrecorder.keys
            model_names = list(self.prior.model_vars)
            point = lambda Θ: {θnm: θval for θnm, θval in zip(optim_names, Θ)}
            model_values = [tuple(self.prior.backward_transform_params(point(Θ)).values())
                            for Θ in optimθrecorder.values]
            assert isinstance(model_values[0][0], np.ndarray)
                # ΘRecorder expects List[Tuple[Array,...]]
            self.Θ_evol = ΘRecorder(**{**optimθrecorder.dict(),
                                      'keys': model_names, 'values': model_values})

    ## Optimizer ##
    @property
    def stepi(self) -> int:
        "Return the optimizer's current/final step."
        # We could load the optimizer from disk and get the actual recorded stepi,
        # but it's more expedient to just use the latest recorded step of any recorder
        return max(rec.steps[-1] for rec in [self.logL_evol, self.Θ_evol, self.latents_evol])

    ## Keys and labels ##
    @property
    def key(self) -> Tuple[str, str, Union[StrTuple,str]]:
        """
        A tuple composed of the hyperparameter label and the fit init key.
        E.g.: ('Λ1', '(5, 10)')
        Combine with a tuple containing the model name and latent variables
        to create a key which is unique across different optimization paradigms.
        Key dimensions can be obtained with `.kdims`
        """
        return self.FitKey(self.Λlabel, self.init_key)
    @property
    def key_label(self) -> str:
        return make_key_label(self.key)
    @property
    def latents(self) -> List[str]:
        return self.latents_evol.keys
    @property
    def init_key(self) -> Union[StrTuple, str]:
        return get_init_key(self.record)
    def ground_truth_key(self, param_index: int=0) -> Tuple[str]:
        return (f'ground_truth-{param_index}',)
    @property
    def hist_index_iter(self) -> Generator[Tuple[str, Tuple[int]]]:
        """
        Yield a sequence of tuples (hist name, hist index).
        Tuples are sorted lexicographically by hist name, then index.
        """
        already_seen_ids = set()
        hists = {}
        # We do two loops in order to sort the history names, and thus
        # ensure consistency in the returned order.
        for hist_name, hist in self.model.nested_histories.items():
            if id(hist) in already_seen_ids:
                # Histories connecting submodels will show up > 1 time.
                continue
            already_seen_ids.add(id(hist))
            hists[hist_name] = hist
        for hist_name in sorted(hists):
            hist = hists[hist_name]
            for hist_index in mtb.utils.index_iter(hist.shape):
                yield hist_name, hist_index

    @property
    def hist_dims(self) -> List[hv.Dimension]:
        """
        Return a list of dimensions for each history component.
        Dimensions are sorted lexicographically as (hist name, hist component).
        """
        return [hist_dims.get(hist_name, hist_index)
                for hist_name, hist_index in self.hist_index_iter]

    ## Model and data ##
    @property
    def model_class(self) -> Type[Model]:
        return get_model_class(self.model_spec)
        # return mtb.iotools._load_types[self.model_name]

    @property
    def model(self) -> Model:
        """
        Return a instance of the inferred model.
        This first time this attribute is accessed, the model is initialized
        with the parameter values at step 0 (before any optimization)."""
        key = self.model_name
        registry = self.model_registry
        if key not in registry:
            model = get_task_param(self.record, 'optimizer.model')
            if isinstance(model, Model):
                registry[key] = model
            elif TaskDesc.looks_compatible(model):
                model = Task.from_desc(model).run()
                assert isinstance(model, Model)
                registry[key] = model
            elif isinstance(model, dict):
                # Assume a serialized Model
                registry[key] = self.model_class.parse_obj(model)
            elif isinstance(model, str):
                # Assume a serialized Model
                registry[key] = self.model_class.parse_raw(model)
            else:
                raise ValueError(f"Unrecognized Model description:\n{model}")
        return registry[key]

    @property
    def prior(self) -> Prior:
        if self._prior is None:
            prior_desc = get_task_param(self.record, "optimizer.prior_params")
            self._prior = Prior.validate(prior_desc)
        return self._prior

    @property
    def data_accessor(self) -> DataAccessor:
        "Return the DataAccessor object used to fit the model."
        data_accessor = get_task_param(self.record, 'optimizer.data_segments.data')
        if isinstance(data_accessor, dict):
            if TaskDesc.looks_compatible(data_accessor):
                data_accessor = Task.from_desc(data_accessor)
                key = data_accessor.digest
                if key in self.data_accessor_registry:
                    # Using a registry avoids re-running the same data-creating task for each fit where it is used.
                    # The (Synthetic)DataAccessor memoizes `load()` calls already
                    data_accessor = self.data_accessor_registry[key]
                else:
                    data_accessor = data_accessor.run()
                    self.data_accessor_registry[key] = data_accessor
            elif DataAccessor.looks_compatible(data_accessor):
                # Use a simple serialization to create keys for data_accessor_registry. Should be unique enough
                key = hash(str(data_accessor))  # Using hash instead of str avoids storing the potentially large str in the dict
                if key in self.data_accessor_registry:
                    data_accessor = self.data_accessor_registry[key]
                else:
                    data_accessor = DataAccessor.from_desc(data_accessor)
                    self.data_accessor_registry[key] = data_accessor
            else:
                raise ValueError("Target data description is not compatible with "
                                 "either Task or DataAccessor serization formats")
        return data_accessor

    @property
    def segment_sampler(self) -> SegmentSampler:
        if self._segment_sampler is None:
            segment_sampler = get_task_param(self.record, 'optimizer.data_segments')
            if isinstance(segment_sampler, dict):
                if TaskDesc.looks_compatible(segment_sampler):
                    # Replace sampler's DataAccessor with one in registry, which is shared with different fits
                    assert 'inputs.data' in segment_sampler
                    segment_sampler['inputs']['data'] = self.target_data
                    # Now create the sampler from the task description
                    segment_sampler = Task.from_desc(segment_sampler)
                    segment_sampler = segment_sampler.run()
                elif 'data' in segment_sampler:
                    segment_sampler['data'] = self.target_data
                    segment_sampler = SegmentSampler(**segment_sampler)
                else:
                    raise ValueError("Segment sampler description is not compatible with "
                                     "either Task or SegmentSampler serization formats")
            elif isinstance(segment_sampler, SegmentSampler):
                # Probably all we need to do is set segment_sampler.data = self.target_data
                # When/if this is needed we'll do that
                raise NotImplementedError("There was no use case during development, "
                                          "but it should not be hard to implement this.")
            else:
                raise NotImplementedError
            self._segment_sampler = segment_sampler
        return self._segment_sampler

    def ground_truth_η(self, trial=None):
        """
        Retrieve the target data for the given trial.
        If no trial is given, the first trial is used.
        """
        if trial is None:
            trial = self.data_accessor.trials.trial.data[0]
        return self.data_accessor.load(trial)

    def ground_truth_Θ(self, param_index: int=0):
        """
        param_index: If there are multiple data sets, this parameter allows
            to specify which one (indexing them sequentially from 0).
        """
        gt_key = self.ground_truth_key(param_index)
        if param_index not in self._ground_truth_Θ:
            synth_data = get_task_param(self.record, "optimizer.data_segments.data")
            if isinstance(synth_data, dict):
                if "taskname" in synth_data:
                    synth_data = smttask.Task.from_desc(synth_data)
                else:
                    synth_data = DataAccessor.from_desc(synth_data)
            assert isinstance(synth_data, (smttask.Task, DataAccessor))
            if isinstance(synth_data, DataAccessor):
                ground_truth = synth_data.trials.trial.data[param_index].params
            elif isinstance(synth_data, sinnfull.tasks.CreateSyntheticDataset):
                #ground_truth = synth_data.param_generator(synth_data.param_keys[param_index]).get_values()
                ground_truth = mtb.typing.IndexableNamespace(
                    **synth_data.prior.random(synth_data.param_keys[param_index]))
            else:  # Any other Task, in particular those loading experimental data
                ground_truth = {}
            if ground_truth:
                normalize = getattr(self.model_class, 'remove_degeneracies', None)
                if normalize:
                    ground_truth = normalize(ground_truth)
                #ground_truth = synth_data.prior.forward_transform_params(ground_truth)
                # Convert to a flattened dict with same keys as θ_curves
                Θidcs = self.Θidcs
                # Casting to np.array ensures this works even with scalar parameters
                # Dict keys must match those of the θ_curves dict
                ground_truth = {
                    (pretty_names.get(θname), StrTuple(θidx)): np.array(ground_truth[θname])[θidx]
                     for θname in Θidcs for θidx in Θidcs[θname]}
            # `gt_key` corresponds to 'init_key',
            self._ground_truth_Θ[gt_key] = ground_truth
        return self._ground_truth_Θ[gt_key]

    @property
    def Θidcs(self):
        return {θname: list(mtb.utils.index_iter(np.array(θval).shape))
                for θname, θval in zip(self.Θ_evol.keys, self.Θ_evol.values[0])}


# %% [markdown]
# ### Plotting fit results

    # %%
    @add_property_to('FitData')
    def logL_curve(self) -> hv.Curve:
        if self._logL_curve is None:
            logL_recorder = self.logL_evol.decimate()
            self._logL_curve = hv.Curve(np.stack((logL_recorder.steps, logL_recorder.values), axis=1),
                                        kdims=['step'], vdims=['log L']
                                       )
        return self._logL_curve

    @add_property_to('FitData')
    def θ_curves(self) -> hv.HoloMap:
        """
        Parameter evolution curves during the fit.

        Returns
        -------
        HoloMap  [θname,θindex]
          :Curve  [step]
        """
        if self._θ_curves is None:
            # Create the list of nd indices for each parameter
            θrecorder = self.Θ_evol
            Θidcs = self.Θidcs
            # Reduce the number of points to plot (the holomaps get heavy)
            # Recorder provides the `decimate` option, but since steps aren't
            # regularly spaced, it's not the most efficient use of points
            # TODO: Option for logspace steps
            nsteps = BokehOpts().θ_curve_points
            steps = θrecorder.steps
            interpolated_steps = np.linspace(steps[0], steps[-1], nsteps)
            # Assemble all the fit traces into a HoloMap
            self._θ_curves = hv.HoloMap(
                {θKey(pretty_names.get(θname), StrTuple(θidx)): hv.Curve(
                     np.stack((interpolated_steps,
                               np.interp(interpolated_steps, steps,
                                         np.array(θrecorder[θname])[(slice(None),)+θidx])),
                              axis=1),
                     kdims=['step'], vdims=[param_dims.get(θname, θidx)],
                     group="Fit dynamics", label=self.make_param_label(θname, θidx)
                    )
                 for θname in Θidcs for θidx in Θidcs[θname]},
                kdims=θKey.kdims,
                label=self.key_label
            )
            self._θ_curves.opts(
                framewise=True,  # Let each parameter set its own range
                axiswise=True
            )
        return self._θ_curves

    @add_to('FitData')
    def get_Θ_at_step(self, step: int) -> list:
        """
        Return the parameter values which were recorded at inference step `step`.
        If parameters were not recorded at that step, return the recorded values
        at the nearest step and print a warning.

        Parameters
        ----------
        step: The inference step for which to retrieve parameters.

        Returns
        -------
        List of parameter values. N-d parameters may be represented by nested
        lists or arrays.

        Warns
        -----
        UserWarning
            : If parameters were not recorded at inference step `step`.
        """
        Θ_step, Θ_evol_index = self.Θ_evol.get_nearest_step(step, return_index=True)
        if Θ_step != step:
            logger.warning(
                f"The parameters were not recorded at step {step}. Using "
                f"the parameters from the closest recorded step: {Θ_step}.")
        Θ_at_step = self.Θ_evol.values[Θ_evol_index]
        # TODO?: Add option to also return Θ_step, Θ_evol_index ?
        return {θname: θval for θname, θval in zip(self.Θ_evol.keys, Θ_at_step)}

    @add_to('FitData')
    def ground_truth_η_curves(self, trial=None):
        if self._gt_η_curves is None:
            bokeh_opts = BokehOpts()
            η = self.ground_truth_η()  # η: xarray.Dataset
            # Reduced number of data points the date so that files aren't so big
            stops = self.model.time.stops_array
            interpolated_stops = np.linspace(stops[0], stops[-1],
                                             bokeh_opts.η_curve_points)
            # Create a Curve for each variable component
            curves = {}
            for hist_name, hist_index in self.hist_index_iter:
                η_trace = η[hist_name]
                stops = η_trace.time.data
                y = η_trace[(slice(None),)+hist_index]
                data = np.stack((interpolated_stops,
                                 np.interp(interpolated_stops, stops, y)),
                                axis=1)
                dim = hist_dims.get(hist_name, hist_index)
                curves[join_keys(self.key, HistKey(hist_name, StrTuple(hist_index)))] = \
                    hv.Curve(data, kdims=['time'], vdims=[dim])

            self._gt_η_curves = hv.HoloMap(curves, kdims=FitData.kdims+HistKey.kdims)
        return self._gt_η_curves

    @add_to('FitData')
    def η_curves(self, step) -> hv.HoloMap:
        """
        Return a HoloMap of latent curves at a particular step.
        Function is cached to allow calling it within multiple DynamicMap functions
        showing different components of the same simulation.

        Returns
        -------
        HoloMap[HistKey]
        """
        bokeh_opts = BokehOpts()
        curves = {}
        model = self.model
        # Create the dimensions and assign them nicely formatted labels (side-effect)
        self.hist_dims
        # Initialize the model with latent histories recorded at `step`
        try:
            step_index = self.latents_evol.steps.index(step)
        except ValueError:
            logger.debug(f"Fit {self} does not contain an η curve at step {step}.")
            # Return a set of blank plots
            for hist in model.history_set:
                for hist_index in mtb.utils.index_iter(hist.shape):
                    dimname = hist.name+''.join(str(i) for i in hist_index)
                    dim = key_dims.get(dimname)
                    panel_key = HistKey(hist.name, StrTuple(hist_index))
                    curves[panel_key] = hv.Curve([], kdims=['time'], vdims=[dim])
        else:
            traces = self.latents_evol.values[step_index]
            for hist_name, trace in zip(self.latents_evol.keys, traces):
                hist = getattr(model, hist_name)
                hist.unlock()
                hist[:len(trace)-hist.pad_left] = np.array(trace)
                    # `hist` expects an AxisIndex, which is the number of bins
                    # after t0idx. Since `trace` also includes the padding bins,
                    # we subtract them
                hist.lock()
            # Set parameters to closest recorded parameters and integrate the model
            # FIXME: Fix Recorder.also_record so we don't have to hide warnings
            # TODO: Allow user to choose to hide warnings when θ and η are not aligned ?
            rootlogger = logging.getLogger()
            logging_level = rootlogger.level
            rootlogger.setLevel(logging.ERROR)
            Θvals = self.get_Θ_at_step(step)
            rootlogger.setLevel(logging_level)
            # Integrate the model with parameters at this step
            model.update_params(Θvals)
            model.clear()
            # The model may be longer than the traces, so we can't just
            # integrate up to 'end' (`integrate` in any case won't integrate
            # beyond the cur_tidx of locked histories, but will display a warning)
            tnidx = max(h.cur_tidx for h in model.history_set)
            model.integrate(upto=tnidx, histories='all')
            # Reduced number of data points the date so that files aren't so big
            tn = model.get_time(tnidx)
            start = getattr(model.t0, 'magnitude', model.t0)
            end = getattr(tn, 'magnitude', tn)
            # Choose a number of points consistent with the resolution of
            # ground_truth_η, which has `η_curve_points` and goes up to model.tn
            n_points = int(round( tnidx/model.time.tnidx*bokeh_opts.η_curve_points  ))
            interpolated_stops = np.linspace(start, end, n_points)
            # Create all the history curves
            for name, hist in model.nested_histories.items():
                trace = hist.get_data_trace()
                times = hist.time_stops
                for hist_index in mtb.utils.index_iter(hist.shape):
                    dimname = name+''.join(str(i) for i in hist_index)
                    dim = key_dims.get(dimname)
                    y = trace[(slice(None),)+hist_index]
                    data = np.stack((interpolated_stops,
                                     np.interp(interpolated_stops, times, y)),
                                    axis=1)
                    # Create the latent hist curve
                    panel_key = HistKey(name, StrTuple(hist_index))
                    curves[panel_key] = hv.Curve(data,  kdims=['time'], vdims=[dim])
        return hv.HoloMap(curves, kdims=HistKey.kdims).opts(framewise=True)


# %% [markdown]
# ### Label & key making methods

    # %%
    @add_to('FitData')
    @staticmethod
    def make_param_label(θname: str, θidx: Union[tuple,str]) -> str:
        idx_str = pretty_names.index_str(θidx)
        return pretty_names.wrap(f"{pretty_names.get(θname)}{idx_str}")


# %% [markdown]
# ## `RSView` definition
#
# Additional attributes w.r.t. *smttask.RecordStoreView*:
#
# - `split_dims`
# - `SplitKey` (Key type for split _children_)
# - `key` (Instance of its _own_ key type, which may be its parent's SplitKey)

# %%
class RSView(smttask.RecordStoreView):

    # Private memoization attributes
    _fitcoll = None

    def __init__(self, *args, key=root_key, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key
        self.split_rsviews = None
        self.split_dims = {}
        self._summaries = None
        self._all_init_keys = None

    @property
    def kdims(self):
        return list(self.split_dims.values())

    def splitby(self, split_fields: Optional[Sequence[str]] = None,
                split_dims: Optional[Sequence[hv.Dimension]] = None,
                drop_unused_split_fields: bool = True,
                get_field_value: Optional[Callable[[Any, str, Any], Any]] = None
               ) -> Dict[Tuple[str], RSView]:
        """
        Specialized `splitby` with default values for split fields and split_dims.
        """
        if self.split_rsviews:
            logger.warning("Replacing existing RecordStoreView split.")
        # Use defaults if no fields are provided
        if split_fields is None:
            split_fields = default_split_fields
        if split_dims is None:
            split_dims = default_split_dims
        split_names = list(split_dims)
        assert len(split_dims) == len(split_fields)  # There must be one dim for each field

        # Split the record store with parent method
        split_rs = super().splitby(
            split_fields=split_fields, split_names=split_names,
            drop_unused_split_fields=drop_unused_split_fields,
            get_field_value=get_field_value)
        # Make the split key values more compact and easier to read
        split_rs = {make_slugs(k): rs for k, rs in split_rs.items()}
        # Assign split views their own key
        for key, rs in split_rs.items():
            rs.key = key

        if not split_rs:
            logger.warning("Cannot split an empty record store view.")
            return self

        # Store some extra attributes
        self.split_rsviews = split_rs
        self.SplitKey = make_key(type(next(iter(split_rs))))
        self.split_dims = {k:v for k,v in split_dims.items() if k in self.SplitKey._fields}
            # Remove dropped fields from split_dims

        # Invalidate pre-computed summaries
        self._summaries = None

        return split_rs

    @functools.lru_cache(maxsize=128)  # lru_cache is not compatible with @add_to decorator
    def get_η_curves(self, splitkey: SplitKey, fitkey: FitData.FitKey, step: int
                    ) -> hv.HoloMap:
        """
        Wrapper around FitData.η_curves to allow memoization
        based on SplitKey and FitKey.
        """
        if self.split_rsviews and splitkey:
            fit_data = self.split_rsviews[splitkey].fitcoll[fitkey]
        else:
            fit_data = self.fitcoll[fitkey]
        return fit_data.η_curves(step)


    # %%
    @add_property_to('RSView')
    def fitcoll(self) -> FitCollection:
        if self._fitcoll is None:
            if self.split_rsviews:
                # If possible, avoid recreating the FitData objects for both parent and split RSView
                self._fitcoll = FitCollection(
                    itertools.chain.from_iterable(
                        rs.fitcoll for rs in self.split_rsviews.values()),
                    key=self.key)
                for split_rs in self.split_rsviews.values():
                    split_rs.fitcoll.key = split_rs.key
            else:
                self._fitcoll = FitCollection(self, key=self.key)
        return self._fitcoll


# %% [markdown]
# ### Record store representation

    # %%
    @add_to('RSView')
    def _repr_mimebundle_(self, include=None, exclude=None):
        if not isinstance(self._iterable, Sequence):
            # Can't compute stats without risking to consume the iterable
            # -> Fall back to parent's repr
            return None
        return hv.Layout([self.counts_table()]
                         + [self.summary_hist(feature)
                            for feature in self.summary_fields]) \
               .cols(1) \
               ._repr_mimebundle_(include, exclude)

# %% [markdown]
# ### Lists of keys
# Convenience properties for different sets of keys. These are most often used to obtain the list of frames for a `HoloMap`.

    # %%
    @add_property_to('RSView')
    def all_fit_keys(self) -> List[FitData]:
        """
        One key per fit.

        Keys: fitcoll key + fit key
        """
        if self.split_rsviews:
            return [join_keys(fitcoll.key, fit.key)
                    for rs in self.split_rsviews.values()
                    for fitcoll in rs.fitcoll.split_by_Λ().values()
                    for fit in fitcoll]
        else:
            return [join_keys(fitcoll.key, fit.key)
                    for fitcoll in self.fitcoll.split_by_Λ().values()
                    for fit in fitcoll]

    @add_property_to('RSView')
    def all_η_steps(self) -> List[int]:
        """
        Return all step values that appear at least once in one of the η fits.
        Steps are sorted in ascending order.
        """
        return sorted(set().union(*(fit_data.latents_evol.steps
                                    for fit_data in self.fitcoll)))

    @add_property_to('RSView')
    def all_η_step_keys(self) -> List[tuple]:
        """
        One key per fit, per recorded log L step; this is the set of frames
        produced by `η_curves`.

        Keys: fitcoll key + fit key + step
        """
        if self.split_rsviews:
            return [join_keys(fitcoll.key, fit.key) + (step,)
                    for rs in self.split_rsviews.values()
                    for fitcoll in rs.fitcoll.split_by_Λ().values()
                    for fit in fitcoll
                    for step in fit.latents_evol.steps]
        else:
            return [join_keys(fitcoll.key, fit.key) + (step,)
                    for fitcoll in self.fitcoll.split_by_Λ().values()
                    for fit in fitcoll
                    for step in fit.latents_evol.steps]

    @add_property_to('RSView')
    def all_init_keys(self):
        """
        Return the set of unique init keys used in any one of the splits.
        The returned keys are sorted, with string keys before tuples.
        """
        if self._all_init_keys is None:
            all_init_keys = set(fit_data.init_key for fit_data in self.fitcoll)
            tup_keys = sorted(k for k in all_init_keys if isinstance(k, tuple))
            str_keys = sorted(k for k in all_init_keys if not isinstance(k, tuple))
            self._all_init_keys = str_keys + tup_keys
        return self._all_init_keys

# %% [markdown]
# ### Whole record store summary info
# The methods below compute summary information on the original (unsplit) RecordStoreView. Each statistic is shown for all splits simultaneously, in one plot or table.
#
# More limited versions of these methods are implemented in smttask.RecordStoreView; the versions below add the following (at the cost of some dependencies to this project):
# - Automatically split histograms if `splitby` was applied.
# - Use this module's `key_dim` registry.
# - Use this module's `root_key` instance.
# - Use this projects plotting configuration (`BokehOpts`)

    # %%
    @add_property_to('RSView')
    def summaries(self):
        if self._summaries is None:
            self._summaries = self.compute_summaries()
        return self._summaries

    @add_to('RSView')
    def compute_summaries(self):
        if self.split_rsviews:
            dframes = {k: rsview.dframe(include=self.summary_fields)
                       for k, rsview in self.split_rsviews.items()}
            hists = {}
            for k, df in dframes.items():
                for field in self.summary_fields:
                    hist = hv.operation.histogram(hv.Table(df[field]), bins='auto')
                    hist = hist.relabel(group=field, label=self.make_split_label(k))
                    assert isinstance(k, tuple)
                    hists[k + (field,)] = hist
            return hv.HoloMap(hists,
                              kdims=list(self.split_dims) + [key_dims.get('rec_stat', label='record statistic')])
        else:
            df = self.dframe(include=self.summary_fields)
            hists = {}
            for field in self.summary_fields:
                hist = hv.operation.histogram(hv.Table(df[field]), bins='auto')
                hist = hist.relabel(group=field, label=root_key.label)
                hists[field] = hist
            return hv.HoloMap(hists, kdims=[key_dims.get('rec_stat', label='record statistic')])

    # %%
    @add_to('RSView')
    def counts_table(self, max_rows=10) -> hv.Table:
        if self.split_rsviews:
            nrows = min(len(self.split_rsviews), max_rows)  # Use a scroll bar if there are too many splits
            table = hv.Table({'RSView': [self.make_split_label(k) for k in self.split_rsviews],
                              'No. records': [len(rsview) for rsview in self.split_rsviews.values()]},
                             kdims=['RSView'], vdims=['No. records'])
        else:
            nrows = 1
            table = hv.Table({'RSView': [root_key.label],
                              'No. records': [len(self)]},
                             kdims=['RSView'], vdims=['No. records'])
        return table.opts(BokehOpts().table(nrows=nrows))

    @add_to('RSView')
    def summary_hist(self, stat_field: str) -> hv.Overlay:
        """
        `feature`: One of the features listed in `self.summary_fields`.
        """
        # Ensure that `feature` matches one of the values
        if stat_field not in self.summary_fields:
            raise ValueError(f"`feature` must be one of {self.summary_fields}. "
                             f"Received {repr(stat_field)}.")
        hists = self.summaries.select(rec_stat=stat_field)
        if isinstance(hists, hv.Histogram):
            hists = [hists]
        hist_opts = BokehOpts().hist_records
        return hv.Overlay([hist.opts(hist_opts) for hist in hists]).collate()

# %% [markdown]
# ### Loading fit data

    # %%
    @add_to('RSView')
    def load_fits(self, Λfields: Optional[Tuple[str]]=None
        ) -> Union[FitCollection,Dict[SplitKey,FitCollection]]:
        """
        Load the fits corresponding to all records.
        If the RSView has been split, fits are returned in a dictionary of
        {split key: fit collection} pairs.
        Otherwise, all fits are returned as one `FitCollection`.

        Parameters
        ----------
        Λfields: Which hyperparameters to include in the fit label.
            See `FitCollection` for more information and default value.

        .. Note:: This function is not cached. For modest data sets, which
           one can be fully loaded in memory, use the `.fitcoll` attribute to
           ensure each fit is loaded only once.
        """
        if Λfields is not None:
            kwargs = {'Λfields': Λfields}
        else:
            kwargs = {}
        if self.split_rsviews:
            return {rskey: FitCollection(rsview, key=key, **kwargs)
                    for rskey, rsview in self.split_rsviews.items()}
        else:
            return FitCollection(self, key=root_key, **kwargs)

# %% [markdown]
# ### Summary info for different hyperparameter sets

    # %%
    @add_property_to('RSView')
    def run_counts_df(self):
        fitcoll = self.fitcoll
        Λdiff_df = fitcoll.Λdiff.dataframe(depth=-1)
        column_levels = ['param group', 'param']  # The returned dframe has a 2-level MultiIndex
        # Determine the column names for each hyperparameter differentiating between sets
        unused_fields = set(fitcoll.Λfields)      # Still print columns for unused fields, with blanks
        param_col_tuples =[]
        for c in Λdiff_df.columns:
            for field in fitcoll.Λfields:
                if c.startswith(field):
                    unused_fields.discard(field)
                    param_col_tuples.append(
                        (pretty_names[field], c[c.index(field)+len(field):].strip('.')))
        # Add columns for differentiating parameters
        if len(param_col_tuples):
            Λdiff_df.columns = pd.MultiIndex.from_tuples(param_col_tuples, names=column_levels)
        else:
            Λdiff_df.columns = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=column_levels)
        # Add 1 column for each non-differentiating parameter field, to show it was checked
        if unused_fields:
            nodiff_columns = pd.MultiIndex.from_tuples([(pretty_names[field], "(No diff)")
                                                        for field in unused_fields])
        else:
            nodiff_columns = pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=column_levels)
        no_Λdiff_df = pd.DataFrame([["-"]*len(unused_fields)]*len(Λdiff_df.index),
                                   index=Λdiff_df.index, columns=nodiff_columns)
        # Add the counts columns – 1 for each split RSView
        if self.split_rsviews:
            split_fitcolls = {split_key: split_rs.fitcoll
                              for split_key, split_rs in self.split_rsviews.items()}
            counts_df = pd.DataFrame(
                {self.make_split_label(split_key, format='latex'):
                    {Λ: len(fitcoll.filter_by_Λ(Λ))
                     for Λ in fitcoll.Λlabels}
                 for split_key, fitcoll in split_fitcolls.items()})
        else:
            # There are no splits – just add one column with all counts
            counts_df = pd.DataFrame(
                {root_key.label:
                    {Λ: len(fitcoll.filter_by_Λ(Λ))
                     for Λ in fitcoll.Λlabels}
                 })
        counts_df.columns = pd.MultiIndex.from_tuples(
            [('# runs', c) for c in counts_df.columns])
        # Return the merged DataFrames
        run_counts_df = pd.concat((Λdiff_df, no_Λdiff_df, counts_df), axis=1)
        return run_counts_df

    # %%
    @add_property_to('RSView')
    def used_init_keys_df(self):
        """
        Construct a boolean `DataFrame` of init_key x split_key.
        True indicates that that combination (init_key, split_key) exists
        If the `RSView` has no splits, the returned `DataFrame` has only one column
        and is True everywhere.
        """
        # Construct the DataFrame
        if self.split_rsviews:
            used_init_keys = pd.DataFrame(np.zeros((len(self.all_init_keys), len(self.split_rsviews)), dtype=bool),
                                          index=self.all_init_keys, columns=list(self.split_rsviews.keys()))
            for split_key, rsview in self.split_rsviews.items():
                init_keys = [fit_data.init_key for fit_data in rsview.fitcoll]
                used_init_keys.loc[init_keys, split_key] = True
        else:
            used_init_keys = pd.DataFrame(np.ones((len(self.all_init_keys), 1), dtype=bool),
                                          index=self.all_init_keys, columns=[root_key.label])
        return used_init_keys

    # %%
    @add_to('RSView')
    def print_used_init_keys(self):
        used_init_keys = self.used_init_keys_df

        key_width = max(len(str(k)) for k in used_init_keys.index) + 1
        max_width = 100
        spacer = "   "
        col_width = key_width + len(used_init_keys.columns) + len(spacer)

        it = iter(used_init_keys.index)
        line = ""
        while True:
            try:
                init_key = next(it)
            except StopIteration:
                if line:
                    print(line)
                break
            if len(line):
                line += spacer
            line += f"{str(init_key):{key_width}}"
            line += ''.join(np.take(["▯","▮"], used_init_keys.loc[[init_key]].values[0]))
            if len(line) + col_width > max_width:
                print(line)
                line = ""

    @add_to('RSView')
    def run_results(self) -> Union[hv.Table,hv.HoloMap]:
        """
        Return tables for each record store split, listing all records with
        their fit parameters, fit result and run durations.
        If the record store view was not split, a single table is returned
        with all records.

        Returns:
        --------
        If RSView was not split, or split into only one subview:
            hv.Table
        If RSView was split into two or more subviews:
            hv.HoloMap
              : hv.Table
        """
        if self.split_rsviews:
            splitkeys = self.split_rsviews.keys()
            fitcolls = [rs.fitcoll for rs in self.split_rsviews.values()]
        else:
            splitkeys = [()]
            fitcolls = [self.fitcoll]

        tables = {}
        for splitkey, fitcoll in zip(splitkeys, fitcolls):
            data = {}
            Λsets = fitcoll.Λsets
            for fit in fitcoll:
                logL = fit.logL_evol.values
                s = fit.record.duration
                h, s = s // 3600, s % 3600
                m, s = s // 60, s % 60
                duration = "{:01}h {:02}m {:02}s".format(int(h),int(m),int(s))
                # TODO: This should be a set of functions outside the loop.
                #   That way: 1. They are easier to modify
                #             2. The special case below where len() == 0
                #                doesn't have to be hard coded.
                reason = fit.record.reason
                if isinstance(reason, tuple):
                    reason = reason[0]
                reason = reason.split('\n')[0]
                data[fit.record.label] = {
                    ('hyperparams', 'init_key'): fit.key.init_key,
                    ('hyperparams', 'Λ set'): fit.key.Λ,
                    ('loss', 'max logL')    : round(max(logL), 2),
                    ('loss', 'last logL')   : round(logL[-1], 2),
                    ('loss', 'has NaNs')    : np.isnan(logL).any(),
                    ('run', 'duration')     : duration,
                    ('run', 'reason')       : reason
                    }
                for Λi_name in fitcoll.abbrev_Λ_names:
                    data[fit.record.label][('hyperparams', Λi_name)] = \
                        Λsets[fit.key.Λ][Λi_name]

            df = pd.DataFrame(
                  data,
                  columns=pd.Index(data.keys(), name='record label'),
                  index=pd.MultiIndex.from_tuples(next(iter(data.values())).keys())
                ).T.sort_values([('loss', 'max logL')], ascending=False)

            # Holoviews Table doesn't currently support multiindex
            df.columns = df.columns.droplevel(0)

            tables[splitkey] = hv.Table(df.reset_index()) \
                .opts(fit_columns=True, width=600, sortable=True, hooks=[hide_table_index])

        if len(tables) == 0:
            # TODO: Get them from the list of display functions
            return hv.Table([], kdims=['Λ set'], vdims=['max logL', 'last logL', 'has NaNs', 'duration', 'reason'])
        elif len(tables) == 1:
            return next(iter(tables.values()))
        else:
            return hv.HoloMap(tables, kdims=self.SplitKey.kdims)

# %% [markdown]
# ### Label & key making methods

    # %%
    @add_to('RSView')
    @staticmethod
    def make_split_label(key: namedtuple, format: str='default') -> str:
        "Overrides `RecordStoreView.make_split_label` to make use of `pretty_names`."
        return make_key_label(key, format)

# %% [markdown]
# ### Plotting fit ensembles

    # %%
    @add_to('RSView')
    def logL_curves(self, color: Optional[ColorEvolCurves]=None,
                    filter: Union[None,str,Callable]=None) -> hv.HoloMap:
        """
        Returns
        -------
        HoloMap (split_key)
         |- NdOverlay (init_key)
             |- Curve (step)
        """
        logL_curves = hv.HoloMap({split_key: rs.fitcoll.logL_curves(color=color, filter=filter)
                                  for split_key, rs in self.split_rsviews.items()},
                                 kdims=list(self.split_dims.values()))
        logL_curves = logL_curves.collate()
        return logL_curves

    # %%
    @add_to('RSView')
    def θ_curves(self, ncols: int=3, width: int=600, color: Optional[ColorEvolCurves]=None,
                 include: Optional[Union[dict,set]]=None,
                 exclude: Optional[Union[dict,set]]=None,
                 ground_truth: Union[bool,hv.Options]=False,
                 dynamic: bool=True
                ) -> hv.NdLayout:
        """
        Arguments `color`, `include` and `exclude` are simply passed on to the
        underlying `FitCollection.θ_curves` calls.

        Parameters
        ----------
        ncols: Numbers of columns to use in layout
        width: Width of the layout, in pixels. Each panel will have
            a width ``width/ncols``.
        color: A `ColorEvolCurves` function which takes an NdLayout and styles
            its curves. See `FitCollection.θ_curves` and `ColorEvolCurves`.
        include: If provided, only these parameter names, or name-index
            combinations, are included.
        exclude: If provided, these parameter names, or name-index combinations,
            are excluded, even if they match an entry in `include`.
        ground_truth: bool | hv.Options
            If True, add horizontal lines to indicate ground truth, using the
            default style for target lines.
            If an 'Options' instance, use that style instead.
            If False, don't draw target lines.
        dynamic: If True, return a DynamicMap. Otherwise return a HoloMap.

        Returns
        -------
        NdLayout (θname, θindex)
         |- Dynamic|HoloMap (split_key)
             |- NdOverlay (init_key)
                 |- Curve (step)
        """
        # TODO: Allow setting defaults
        panel_width = int(width//ncols)
        panel_height = panel_width
        if not self.split_rsviews:
            fitcolls = [self.fitcoll]
            layout = self.fitcoll.θ_curves()
        else:
            # We used to have an implementation where we created nested all the NdLayout,
            # and used .collate() to merge the HoloMaps, but because of the large number
            # of curves the collation was quite slow.
            # We have enough structure to do the collation ourselves, which is much faster.
            all_θ_layouts = [rs.fitcoll.θ_curves(color=color, include=include, exclude=exclude)
                             for rs in tqdm(self.split_rsviews.values(),
                                            desc="Overlaying ensembles of θ curves")]
            # HoloMaps behave poorly with missing data => create a set of
            # blank panels with the same kdims, vdims, etc. (for the widget
            # to work correctly, it is essential that dimensions match)
            blanks = {}
            for θ_layout in all_θ_layouts:
                for panelkey, hm_panel in θ_layout.items():
                    if panelkey not in blanks:
                        ndoverlay = next(iter(hm_panel))
                        assert isinstance(ndoverlay, hv.NdOverlay)
                            # If we need other types, we can probably add some if clauses
                        curve = next(iter(ndoverlay))
                        blank_frame = next(iter(ndoverlay)).clone(
                            [curve.clone([(0,0), (0,0.1)])])
                        blanks[panelkey] = ndoverlay.clone({'None': blank_frame})
            # Set of parameters may differ between splits => Obtain a list of all θ keys
            panel_keys = sorted(set().union(*(θlayout.keys() for θlayout in all_θ_layouts)))
            # The different splits may also have different Λ keys
            frame_keys = set().union(*(holomap.keys() for layout in all_θ_layouts for holomap in layout))
            frame_keys = {wrap_with_tuple(key) for key in frame_keys}  # Allow us to assume below that keys are tuples
            frame_kdims_it = (holomap.kdims for layout in all_θ_layouts for holomap in layout)
            frame_kdims = next(frame_kdims_it)
            assert all(kdims == frame_kdims for kdims in frame_kdims_it)
            # Extend frame_keys to full outer product, b/c HoloMap widget assumes this
            key_len = len(next(iter(frame_keys)))
            assert all(len(key) == key_len for key in frame_keys)
            frame_keys = list(itertools.product(*(set(key[i] for key in frame_keys) for i in range(key_len))))
            # However, the kdims must be the same
            layout_kdims = all_θ_layouts[0].kdims
            assert all(layout.kdims == layout_kdims for layout in all_θ_layouts[1:])
            layout_data = {}
            # Populate layout_data by merging the holomaps of each fitcoll
            with TimeThis("Merge fitcoll HoloMaps"):
                for panelkey in panel_keys:
                    # For this panel, combine all ensemble overlays into one HoloMap
                    # # Not all θ may be in each layout, so the list below may be shorter than len(all layouts)
                    #coll_frames = [layout[panelkey] for layout in all_θ_layouts if panelkey in layout]
                    #coll_kdims = coll_frames[0].kdims
                    #assert all(frame.kdims == coll_kdims for frame in coll_frames)
                    frame_data = {}
                    for framekey in frame_keys:
                        # We need one value per frame key
                        # Assign the blank default, and then loop over all holomaps
                        # at for this panel, looking for one matching framekey.
                        # If one is found, it is unique, so replace default and break
                        frame_data[framekey] = blanks[panelkey].clone()
                        for θ_layout in all_θ_layouts:
                            if panelkey in θ_layout and framekey in θ_layout[panelkey]:
                                frame_data[framekey] = θ_layout[panelkey][framekey]
                                break
                    layout_data[panelkey] = hv.HoloMap(
                        # FitCollection.θ_curves already augments keys with split dims
                        frame_data,
                        kdims=frame_kdims
                    )
            layout = hv.NdLayout(layout_data, layout_kdims)
        θ_curves = layout.cols(ncols).opts(hv.opts.Curve(width=panel_width, height=panel_height))

        if ground_truth:
            if ground_truth is True:
                ground_truth = None  # Use default value in `add_ground_truth_targets`
            with TimeThis("Add ground truth targets"):
                θ_curves = self.add_ground_truth_targets(θ_curves, ground_truth)

        if dynamic:
            with TimeThis("Make dynamic"):
                #θ_curves = θ_curves.clone(
                #    {panelkey: hv.util.Dynamic(holomap_panel)
                #     for panelkey, holomap_panel in θ_curves.items()})
                θ_curves = hv.NdLayout(
                    {panelkey: hv.util.Dynamic(holomap_panel)
                     for panelkey, holomap_panel in θ_curves.items()},
                    kdims = θ_curves.kdims)

        return θ_curves


    # %%
    # TODO: Move core functionality to FitCollection ?
    @add_to('RSView')
    def add_ground_truth_targets(
        self, θ_curves: hv.Layout, hline_style: Optional[hv.Options]=None
        ) -> hv.NdLayout:
        """
        Overlay HLine objects onto a set of panels to indicate ground truth values.

        If the RSView has been splitted, the returned layout has separate frames
        for both different splits and different Λ.
        If the RSView has been splitted, the returned layout has separate frames
        only for different Λ.

        Returns
        -------
        A new Layout object, with similar hierarchical structure as `θ_curves`.
        """
        if hline_style is None:
            hline_style = BokehOpts().target_hline

        if (not isinstance(θ_curves, (hv.Layout, hv.NdLayout))
            or not isinstance(next(iter(θ_curves)), hv.HoloMap)):
            raise TypeError("`θ_curves` must be a Layout [θname, θkey] containing "
                            f"HoloMap objects. Received:\n{θ_curves}.")
        if self.split_rsviews:
            fitcolls = [fitcoll
                        for rs in self.split_rsviews.values()
                        for fitcoll in rs.fitcoll.split_by_Λ().values()]
        else:
            fitcolls = [fitcoll for fitcoll in self.fitcoll.split_by_Λ().values()]
        targets = {join_keys(fitcoll.key,θKey(*θkey)): θval
                   for fitcoll in fitcolls
                   for θkey, θval in fitcoll.ground_truth().items()}

        res = {}
        for θkey, holomap in θ_curves.items():
            for holokey, overlay in holomap.items():
                try:
                    target_line = hv.HLine(
                        targets[wrap_with_tuple(holokey)+wrap_with_tuple(θkey)], group="ground truth"
                        ).opts(hline_style)
                    # FIXME: This should use `join_keys`
                    res[wrap_with_tuple(θkey)+wrap_with_tuple(holokey)] = overlay * target_line
                except KeyError:
                    # Still replace NdOverlay by Overlay => HoloMap requires all elements of same type
                    # And same depth ! -> https://github.com/holoviz/holoviews/issues/3963
                    # See also https://github.com/holoviz/holoviews/issues/4229
                    # Create a blank target line which won't modify the yaxis range
                    curve = next(iter(overlay)); assert isinstance(curve, hv.Curve)
                    range_x = curve.range(curve.dimensions()[0]); Δx = np.diff(range_x)
                    range_y = curve.range(curve.dimensions()[1]); Δy = np.diff(range_y)
                    if not np.isfinite(range_y).all():
                        range_y = (0, 1)
                    #blank_target_line = hv.HLine(
                    #    np.mean(range_y), group="ground truth"
                    #    ).opts(line_width=4)
                    blank_target_line = hv.Curve([(range_x[0],range_y[0]),
                                                  (range_x[0]+0.01*Δx, range_y[0]+0.01*Δy)],
                                                 group="ground truth"
                                                ).opts(alpha=0.1)
                    res[θkey+holokey] = overlay * blank_target_line
        return hv.HoloMap(
            res, kdims=θ_curves.kdims+holomap.kdims
        ).opts(axiswise=True, framewise=True).layout(θKey.kdims)

# %% [markdown]
# ### Plotting latents

    # %%
    @add_to('RSView')
    def η_curves(self, init_fit: Optional[FitData]=None,
                 init_step: Optional[int]=None, ncols: int=4) -> hv.Layout:
        """
        Parameters
        ----------
        init_fit: The fit to show initially before widgets are modified.
            The panels (history name & index) are inferred from this fit.
            If not provided, the first fit in the RSView is used.
        init_step: The initial fit step to show. Defaults to the last step.
        ncols: Number of panel columns to show.

        Returns
        -------
        hv.Layout[HistKey]
        """
        bokeh_opts = BokehOpts()
        if init_fit is None:
            init_fit = next(iter(self.fitcoll))
        if init_step is None:
            init_step = init_fit.latents_evol.steps[-1]
        # Workaround: Recover the split_key for the split containing `init_fit`
        if not self.split_rsviews:
            # RSView was not split – use it instead of its splits
            init_split_key = self.fitcoll.key
        else:
            for split_key, rs in self.split_rsviews.items():
                if init_fit in rs.fitcoll:
                    init_split_key = split_key
                    break
            else:
                init_split_key = next(iter(self.split_rsviews.keys()))
                logger.error(f"No split contains the fit {init_fit}. "
                             f"Defaulting to split '{init_split_key}'.")
        ## Enumerate all possible values for the key dimensions
        # First enumerate values for SplitKey and FitKey
        if self.split_rsviews:
            merged_keys = [join_keys(splitkey,fitkey)
                           for splitkey, rs in self.split_rsviews.items()
                           for fitkey in rs.fitcoll.keys()]
            kdims = self.SplitKey.kdims + FitData.kdims
        else:
            merged_keys = list(self.fitcoll.keys())
            kdims = FitData.kdims[:]  # Make a copy because we will append
        # TODO: Use sinnfull.utils.get_field_values
        kvalues = {dim.name: sorted(set(key[i] for key in merged_keys))
                   for i, dim in enumerate(kdims)}
        kdefaults = {field: val for field, val in zip(init_split_key._fields, init_split_key)}
        kdefaults.update({field: val for field, val in zip(init_fit.key._fields, init_fit.key)})
        # Then enumerate step values
        kdims.append(key_dims.get('step'))
        kvalues['step'] = self.all_η_steps
        kdefaults['step'] = init_step
        ## Create new key dimensions, setting `values` and `default`.
        #  This leaves the original dimensions untouched
        kdims = [hv.Dimension(dim.name, label=dim.label,
                              values=kvalues[dim.name], default=kdefaults[dim.name])
                 for dim in kdims]
        ## Create the initial HoloMap, and construct panels
        #  (one panel per history component, each composed of a DynamicMap)
        init_map = init_fit.η_curves(init_step)
        panels = []
        split_len = len(self.kdims)
        fit_len = len(FitData.FitKey._fields)
        # for component_idx in init_map.keys():
        for histnm, histidx in init_fit.hist_index_iter:
            def dyn_wrapper(*args, idx=(histnm,StrTuple(histidx))):
                # Unpack *args into SplitKey, FitKey, step
                split_key, args = args[:split_len], args[split_len:]
                fit_key, args = args[:fit_len], args[fit_len:]
                assert len(args) == 1
                step = args[0]
                # Retrieve the HoloMap, and index the desired component
                # (the HoloMap is only created once for all components,
                # thanks to caching of RSView.η_curves)
                cur_η = self.get_η_curves(split_key, fit_key, step)[idx].opts(framewise=True)
                # Retrieve the target (true) latent trace
                if self.split_rsviews and split_key:
                    fit = self.split_rsviews[split_key].fitcoll[fit_key]
                else:
                    fit = self.fitcoll[fit_key]
                true_η = fit.ground_truth_η_curves()[fit_key+idx]  # FIXME: At present we always use default trial
                # Convert to a filled curve (aka area)
                true_η = true_η.to.area().opts(bokeh_opts.true_η).opts(framewise=True)
                return (true_η * cur_η).opts(framewise=True)
            panels.append(hv.DynamicMap(dyn_wrapper, kdims=kdims).opts(framewise=True))

        ## Arrange panels into a Layout
        return hv.Layout(panels).cols(ncols)

# %% [markdown]
# ### Example: Record store overview

# %% [markdown]
# In a notebook, `RSView` has a rich representation listing number of records, record timestamps and durations. If the RSView has not been split, these statistics are shown for all records together.

# %% [markdown]
# ```python
# rsview = RSView().filter.tags('finished') \
#          .filter.after(20201212).filter.before(20201223) \
#          .filter.reason("Model & objective selection") \
#          .list
# rsview.add_tag('demo_fits')
# ```

# %%
if __name__ == "__main__":
    rsview = RSView().filter.tags('demo_fits').list
    display(rsview)

# %% [markdown]
# After splitting an `RSView`, its representation changes to show statistics for each split RSView separately.

# %%
if __name__ == "__main__":
    split_rs = rsview.splitby()
    display(rsview)


# %% [markdown]
# ## `FitCollection` definition
# A loose collection of `FitData` objects. All information is stored in the `FitData` objects (record, hyperparams), so a collection can group them in whichever way makes most sense for the needs of the plot or analysis.

# %%
class FitCollection(Dict[tuple, FitData]):
    """
    A loose collection of FitData objects. All information is stored in the
    FitData objects (e.g. record, hyperparams), so a collection can group
    them in whichever way makes most sense for the needs of the plot or analysis.
    """
    def __init__(self, fits: Iterable[Union[RecordView, FitData]],
                 key: tuple,
                 Λfields=('optimizer.fit_hyperparams',)):
        """
        Parameters
        ----------
        fits: List of either `Record` or `FitData` objects. If records,
            they must correspond to a fitting task (specifically, define
            output data paths for 'log L', 'Θ' and 'latents').
        key: A unique key identifying this collection. E.g. if the collection
            was produced from a splitted RSView, the associated 'split_key'
            would be a good choice.
        Λfields: Each FitData instance has a Λlabel attribute indicating
            which hyperparameters were used to produce it. `Λfields` indicates
            which Record fields should be used to determine this label.
            This parameter is ignored for elements of `fits` which are FitData
            instances, since those already have a Λlabel.
        """
        if not isinstance(key, str) and not hasattr(key, '_fields'):
            raise TypeError("`key` must be a namedtuple (or at least define '_fields').")
        super().__init__()  # Initialize 'self' as an empty list
        self.key = key
        self.Λfields = Λfields
        self._Λdiff = None
        self._Λdiff_digest = None  # Used to detect if Λ_comparison is stale
        # self.abbrev_Λ = None
        self._logL_curves = None
        self._logL_curves_keys = None
        self.add_fits(fits)

    def __iter__(self):
        return iter(self.values())

    def __getitem__(self, key):
        """Also allow selecting with the collection's own key prepended."""
        if isinstance(key, tuple) and key[:len(self.key)] == self.key:
            return super().__getitem__(key[len(self.key):])
        else:
            return super().__getitem__(key)

    def __contains__(self, key_or_fit: Union[FitData,FitData.FitKey]):
        if isinstance(key_or_fit, FitData):
            # Match by ID to avoid comparing fields
            ids = [id(fit) for fit in self]
            return id(key_or_fit) in ids
        else:
            return key_or_fit in self.keys()

    def add_fits(self, fits: Iterable[Union[RecordView, FitData]]):
        """
        Add the fits in `fits` to the `FitCollection`.
        If a fit with the same key is already present, the new fit is only
        added if its `stepi` attribute is higher. In this case it replaces
        the previous fit.
        """
        for fit in fits:
            # TODO: Don't create Λregistry entries on a failing execution,
            #       without using an initial loop which would consume the
            #       `fits` iterator. Maybe Λregistry could have a 'roll back' ?
            if not isinstance(fit, (FitData, RecordView)):
                raise TypeError("A FitCollection must be initialized with a "
                                "an iterable of FitData or Record instances. "
                                f"Encountered an entry of type {type(fit)}.")
            if isinstance(fit, FitData):
                if fit.key in self:
                    if fit.stepi > self[fit.key].stepi:
                        self[fit.key] = fit
                else:
                    self[fit.key] = fit
                continue

            assert isinstance(fit, RecordView)
            record = fit
            if any(nm not in ''.join(record.outputpaths) for nm in ['log L', 'Θ', 'latents']):
                logger.warning(f"Record {record.label} does not point to output files "
                               "with names expected for a fit run: 'log L', 'Θ' and 'latents'.\n"
                               "Skipping.")
                continue
            param_set = ParameterSet({Λname: get_task_param(record, Λname, '<No value>')
                                      for Λname in self.Λfields})
            Λlabel = Λregistry.get_label(param_set)

            fit_data = FitData(Λ     =param_set,
                               Λlabel=Λlabel,
                               record=record)
            if fit_data.key in self:
                if fit_data.stepi > self[fit_data.key].stepi:
                    self[fit_data.key] = fit_data
            else:
                self[fit_data.key] = fit_data

        # Check that all FitData objects share the same data_accessor registry
        # TODO?: Don't print warning if model_names differ ?
        if len(set(id(fit_data.data_accessor_registry)
                   for fit_data in self)) > 1:
            logger.warning("Not all fits are using the same DataAccessor; this "
                           "may cause a ground truth model to be unnecessarily "
                           "integrated multiple times.")

    @property
    def key_names(self) ->Tuple[str]:
        if isinstance(self.key, str):
            return ()
        else:
            return self.key._fields
    @property
    def key_label(self) -> str:
        "Merge the key values and names together into a readable string"
        return make_key_label(self.key)

    @property
    def kdims(self) -> List[hv.Dimension]:
        return [key_dims.get(nm) for nm in self.key_names]
    @property
    def Λlabels(self) -> set:
        return set(fit.Λlabel for fit in self)
    @property
    def Λdigests(self) -> set:
        Λlabels = self.Λlabels
        return set(d for d, lbl in Λregistry.labels.items() if lbl in Λlabels)
    @property
    def Λsets(self) -> dict:
        return {Λname: Λregistry.get_param_set(Λname) for Λname in self.Λlabels}

    @property
    def Λdiff(self) -> ParameterComparison:
        cmp_digest = ''.join(sorted(self.Λdigests))  # Testing the digest makes this robust against
        if cmp_digest != self._Λdiff_digest:         # fits added with plain list methods
            Λsets = self.Λsets
            self._Λdiff = ParameterComparison(Λsets.values(), list(Λsets.keys()))
            self._Λdiff_digest = cmp_digest
        return self._Λdiff
    @property
    def abbrev_Λ_names(self) -> List[str]:
        """Return the list of hyperparameter names which differ between fits."""
        return list(self.Λdiff.comparison.keys())


# %% [markdown]
# ### Filtering a fit collection

    # %%
    @add_to('FitCollection')
    def filter_by_init_key(self, init_key: Union[StrTuple,str]):
        matching_fits = []
        for fit in self:
            if fit.init_key == init_key:
                matching_fits.append(fit)
        return FitCollection(matching_fits, key=self.key, Λfields=self.Λfields)

    @add_to('FitCollection')
    def filter_by_Λ(self, Λ: Union[str,ParameterSet]) -> FitCollection:
        """
        Return a new FitCollection, composed only of fits matching the given
        hyperparameter.

        Parameters
        ----------
        Λ: If a str, assumed to be a parameter set _label_.
           If a ParameterSet, the Λregistry is queried for the associated label.

        Returns
        -------
        A fit FitCollection which is a subset of self.
        """
        if isinstance(Λ, ParameterSet):
            Λlabel = Λregistry.get_label(Λ, create_if_missing=False)
        elif isinstance(Λ, str):
            Λlabel = Λ
        else:
            raise TypeError("`Λ` must either be a ParameterSet, or parameter set label.\n"
                            f"Received {Λ} (type: {type(Λ)})")
        matching_fits = []
        for fit in self:
            if fit.Λlabel == Λlabel:
                matching_fits.append(fit)
        return FitCollection(matching_fits, key=self.key, Λfields=self.Λfields)

    # %%
    @add_to('FitCollection')
    def split_by_Λ(self) -> Dict[str,FitCollection]:
        """
        Split a FitCollection into disjoint sub collections such that all fits
        in a sub collection share the same hyperparameters.

        Returns
        -------
        Dictionary of {hyperparam label: sub FitCollection} pairs.
        """
        if 'Λ' in self.key._fields:
            logger.warning("The FitCollection is already split by Λ. Nothing to do.")
            return self  # EARLY EXIT
        KeyType = namedtuple(type(self).__name__+"_Λ", self.key._fields + ('Λ',))
        sub_colls = {}
        for fit in self:
            key=KeyType(*self.key, fit.Λlabel)
            if key not in sub_colls:
                sub_colls[key] = FitCollection([fit], key=key)
            else:
                sub_colls[key].add_fits([fit])
        return sub_colls


# %% [markdown]
# ### Aggregating statistics / data

    # %%
    @add_to('FitCollection')
    def ground_truth(self):
        """
        Return a dictionary of ground truth values, but only if the same
        ground truth is shared by all fits in the collection.
        """
        it = iter(self.values())
        gt = next(it).ground_truth_Θ()
        for fit_data in it:
            if gt != fit_data.ground_truth_Θ():
                logger.warning(f"The fits in FitCollection {self.key} "
                               "don't all have the same ground truth.")
                gt = {}
                break
        return gt

# %% [markdown]
# ### Plotting fit ensembles

    # %%
    @add_to('FitCollection')
    def logL_curves(self, color: Optional[ColorEvolCurves]=None,
                    filter: Union[None,str,Callable]=None) -> hv.HoloMap:
        """
        Obtain each logL curve and combine them so that curves differing
        only by their init_key are plotted together (i.e. in the same frame)

        Parameters
        ----------
        color_curves: Whether to apply the method `color_evol_frames` to
            colour evols with the highest log L with an accent colour.

        filter: None | str | Callable
            None: All curves are returned
            Callable: Signature must be (fit_data) -> bool
                Only fits for which this returns True are returned
            str: Some common filters are named for convenience.
                - 'nofail': Remove fits for which the outcome was 'Failed'
            If provided, only fits satisfying this condition are returned.

        Returns
        -------
        HoloMap(fitcoll.key, curve.key)
         |-- Overlay(init_key)
              |-- Curve
        """
        if not self._logL_curves or self._logL_curves_key_set != set(self.keys()):

            logL_curves = hv.HoloMap(
                {join_keys(self.key, fit_data.key): fit_data.logL_curve for fit_data in self},
                kdims=self.kdims+FitData.kdims)
            frames = logL_curves.overlay('init_key')

            for frame in frames:
                clip_yrange(frame, 0.05, 1)

            self._logL_curves_key_set = set(self.keys())  # Detect if logL cache is stale
            self._logL_curves = frames
        else:
            frames = self._logL_curves

        if filter:
            # Common named filters
            if isinstance(filter, str):
                if filter == "nofail":
                    filter = lambda fit_data: "<OptimizerStatus.Failed>" not in fit_data.record.outcome
                else:
                    raise ValueError(f"Filter name '{filter}' not recognized.")
            # Keep only curves satisfying the filter condition
            init_keys = [fit_data.key.init_key for fit_data in self if filter(fit_data)]
            frames = frames.select(init_key=init_keys)

        if color:
            color(frames)

        return frames


    # %%
    @add_to('FitCollection')
    def θ_curves(self, color: Optional[ColorEvolCurves]=None,
                 include: Optional[Union[dict,set]]=None,
                 exclude: Optional[Union[dict,set]]=None) -> hv.NdLayout:
        """
        Return a HoloMap of all θ evolutions.

        In contrast to `RSView.θ_curves`, a 'dynamic' flag is not relevant
        because all curves are rendered on the same frame.

        .todo:: Fill in missing keys with blank plots.

        Parameters
        ----------
        color: A `ColorEvolCurves` function which takes an NdLayout and styles
            its curves. See `FitCollection.θ_curves` and `ColorEvolCurves`.
        include: If provided, only these parameter names, or name-index
            combinations, are included.
        exclude: If provided, these parameter names, or name-index combinations,
            are excluded, even if they match an entry in `include`.

        Returns
        -------
        NdLayout [θname,θindex]
         |- HoloMap [model,latents,Λ]
             |- NdOverlay [init_key]
                 |- Curve [step]
        """
        # Each fit_data.θ_curves is a HoloMap.
        # To merge them, we need to augment each HoloMap key with the
        # fitcoll and fit_data keys, so we create a new dictionary and then
        # recreate the HoloMap
        θ_curves = {}
        for fit_data in self:
            for curve_key, curve in fit_data.θ_curves.items():
                curve_key = θKey(*curve_key)  # Cast back to θKey to reattach _fields
                θ_curves[join_keys(self.key, fit_data.key, curve_key)] = curve

        with TimeThis("Create HoloMap"):
            θ_curves = hv.HoloMap(θ_curves, kdims=self.kdims+FitData.kdims+θKey.kdims,
                                  group=curve.group, label=make_key_label(self.key))

        if isinstance(include, (list,set)):
            θ_curves = θ_curves.select(θname=include)
        elif isinstance(include, dict):
            tmp_map = hv.HoloMap(kdims=θ_curves.kdims, group=θ_curves.group, label=θ_curves.label)
            for θname, θidx in include.items():
                θsel = θ_curves.select(θname=θname, θindex=θidx)
                if isinstance(θsel, hv.HoloMap):
                    for key, el in θsel.items():
                        tmp_map[key] = el
                else:
                    raise NotImplementedError
            θ_curves = tmp_map
        elif include is not None:
            raise TypeError("`include` must be either a set or dict, or None. "
                            f"Received {include} (type: {type(include)}).")

        if isinstance(exclude, (list,set)):
            θnames = set(θ_curves.dimension_values('θname')) - set(exclude)
            θ_curves = θ_curves.select(θname=θnames)
        elif isinstance(exclude, dict):
            tmp_map = hv.HoloMap(kdims=θ_curves.kdims, group=θ_curves.group, label=θ_curves.label)
            θname_dimidx = θ_curves.get_dimension_index('θname')
            θidx_dimidx = θ_curves.get_dimension_index('θindex')
            for key, el in θ_curves.items():
                θname = key[θname_dimidx]
                if θname not in exclude:
                    tmp_map[key] = el
                else:
                    θidx = key[θidx_dimidx]
                    if θidx not in exclude[θname]:
                        tmp_map[key] = el
            θ_curves = tmp_map
        elif exclude is not None:
            raise TypeError("`exclude` must be either a set or dict, or None. "
                            f"Received {exclude} (type: {type(exclude)}).")

        with TimeThis("Create overlays"):
            axes = θ_curves.overlay('init_key').opts(axiswise=True, framewise=True)

        # TODO: Fill in missing frame keys with blank plots.
        #       Compute outer product of keys => data={}, loop => if missing, blank_overlay => hv.HoloMap(data)

        with TimeThis("Clip yranges"):
            for ax in axes:
                clip_yrange(ax, 0.05, 0.95)

        with TimeThis("Color curves"):
            if color:
                color(axes)

        with TimeThis("Make layout"):
            panels = axes.layout(['θname', 'θindex']).opts(axiswise=True, framewise=True)

        for key, panel in panels.items():
            label = param_dims.get(*key).label
            panel.relabel(label=label)
            panel.opts(title=label)

        return panels


# %% [markdown]
# ## Examples

# %% [markdown]
# ### Example: Listing hyperparameter combinations, number of runs and init keys.
# Display a DataFrame showing each hyperparameter ($\Lambda$) set, which parameters differ between sets, and how many records each split RSView ran with each $\Lambda$ set.
#
# Below this, print the list of RNG keys used to initialize the fits. To the right of each key are a set of boxes representing models-hyperparameter combinations; filled boxes indicate that that combination was fit with that initialization (so 3 models + 4 different learning parameter combinations could produce up to 12 box columns columns, if each combination was tried with each model).
# To determine which model & hyperparameters correspond to each rectangle column, list the non-empty combinations from the run totals DataFrame, first top to bottom than left to right (so combinations with the same model are grouped together).

# %%
if __name__ == "__main__":
    display(rsview.run_counts_df)
    rsview.print_used_init_keys()

# %% [markdown]
# ### Example: Plotting fit dynamics
#
# > **Note:** This example uses a HoloViews `DynamicMap`. To interact with it in a JupyterLab notebook, you need to have [installed the PyViz JupyterLab extension](http://holoviews.org/#installation). The widget may also get confused if you reopen the notebook in another window or tab (symptom: changing values has no effect). This can be fixed by shutting down the kernel (NOT simply restarting), closing the notebook tab and reopening again.<br>
# > To produce `HoloMap` based version suitable embedding into a static HTML document, pass `dynamic=False`. This takes more time to produce the figure, since every frame must be created before it is shown.

# %%
if __name__ == "__main__":
    color_fn = ColorEvolCurvesByMaxL(rsview.logL_curves(), quantile=.3, window=1000)
    display(rsview.logL_curves(color=color_fn))
    display(rsview.θ_curves(exclude={'M', 'Mtilde'}, color=color_fn, ground_truth=True))

# %% [markdown]
# ### Example: Plotting inferred latents
#
# > **Note:** This example uses a HoloViews `DynamicPlot`. The same remarks made above apply.<br>
# > In contrast to `θ_curves`, `η_curves` does not provide a `dynamic=False` option – with even modest numbers of fits (3 x 22), a static HoloMap version can take > 30 min to produce (because there is one frame for each step). To produce a static version, use the mechanisms provided by HoloViews for converting a `DynamicMap` to a `HoloMap`, and consider restricting the set of key values.

# %%
if __name__ == "__main__":
    # Choose the best fit to show first
    fitcoll = rsview.split_rsviews[('OUInput', 'Ĩ')].fitcoll
    fits = {get_logL_quantile(fit.logL_curve.data, quantile=0.3, window=1000): fit
            for fit in fitcoll}
    init_fit = fits[max(fits)]
    display(rsview.η_curves(init_fit).cols(2))

# %% [markdown] tags=["remove-cell"]
# ## Wrap-up
# Call `update_forward_refs` on the Pydantic models to ensure all the types they depend on are defined.

# %% tags=["remove-cell"]
FitData.update_forward_refs()

# %%
