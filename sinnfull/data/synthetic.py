# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
# ---

# %% [markdown]
# # Synthetic Data Accessor
#
#
# Instead of retrieving data from disk, the synthetic data accessor simulates a model with certain parameters. Thus, instead of a *datadir* parameter, it is specified by a *model*, a list of $N_{\mathrm{trials}}$ *parameter sets* and a list of $N_{\mathrm{trials}}$ *simulation keys*; the simulation keys are used to set the simulation RNG to a reproducible state. When the `load()` method is called, an integer is uniformly drawn from $[0, N_{\mathrm{trials}})$ and the parameter set and simulation key at the corresponding index selected. The model and RNG are initialized with these parameters and integrated to generate the required data.
#
# All simulation keys should be different, but repeated parameter sets can be used to specify multiple realizations of the same model.
#
# The *xarray* `Dataset` returned by `load()` contains one `DataArray` for each unlocked
# history in the model.

# %%
from __future__ import annotations

# %%
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup('theano')
    #%matplotlib widget

# %%
import os
import re
import warnings
from warnings import warn
from typing import Optional, Union, List, Tuple, Dict
from types import SimpleNamespace
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray, Dataset
from pydantic import BaseModel, PrivateAttr, validator, root_validator
from pydantic.typing import ClassVar
import pydantic.parse
from sinn.utils.pydantic import add_exclude_mask

from sinnfull.parameters import ParameterSet
from sinnfull.typing_ import IndexableNamespace

import mackelab_toolbox as mtb
import mackelab_toolbox.iotools
from mackelab_toolbox.typing import Array
import mackelab_toolbox.utils

# %% [markdown]
# ## Definition

# %%
import theano_shim as shim
from typing import Sequence, Iterable
from mackelab_toolbox.typing import json_like
from sinn.models import Model, ModelParams
from sinnfull.data.base import DataAccessor as BaseAccessor, Trial as BaseTrial
import sinnfull.utils as utils

# %%
__all__ = ["SyntheticTrial", "SyntheticDataAccessor"]

# %%
class SyntheticTrial(BaseTrial):
    # Inherited from BaseTrial:
    # - Config: extra='forbid'
    # - __hash__
    model    : Optional[Model]=None  # Only used for initialization; not exported by default
    params   : IndexableNamespace
    init_cond: IndexableNamespace
    seed     : int
    keynames : ClassVar[Tuple[str,str,str]]=('Θ_hash', 'ic_hash', 'seed')
        
    @root_validator(pre=True)
    def params_from_model(cls, values):
        params = values.get('params')
        model = values.get('model')
        # If needed, use `model` to construct `params` from a dictionary
        if isinstance(params, (dict,IndexableNamespace)):
            if model is None:
                raise ValueError("If `params` is passed as a dictionary, "
                                 "`model` must also be provided.")
            else:
                params = model.Parameters(**params)
            values['params'] = params
        return values

    @validator('params', 'init_cond', pre=True)
    def fix_params(cls, ns):
        """
        Convert possibly symbolic/shared parameters to numeric values.
        This prevents them from being modified elsewhere.
        """
        if isinstance(ns, ModelParams):
            ns = ns.get_values()
        return ns
            
    @validator('params', 'init_cond', pre=True)
    def flatten_nested_params(cls, ns):
        """
        Standardize the form of `params` and `init_cond` so that code doesn't
        need to worry about hierarchies.
        """
        # Especially if the parameters are being deserialized from smt record
        # parameters, the lower levels can be array objects
        # {'submodel': array({'subhist': ['Array', …]}, dtype=object) }
        new_ns = {}
        for k, v in ns.items():
            if isinstance(v, np.ndarray) and v.dtype is np.dtype('O'):
                v = v[()]
            if isinstance(v, dict):
                v = cls.flatten_nested_params(v)
                for subk, subv in v.items():
                    new_ns[f"{k}.{subk}"] = subv
            else:
                new_ns[k] = v
        return new_ns
        
    @validator('params', 'init_cond', pre=True)
    def deserialize_arrays(cls, ns):
        """
        Deserialize Arrays manually, since IndexableNamespace does not do it
        """
        for key, val in ns.items():
            if isinstance(val, SimpleNamespace):
                # Recurse into parameters for nested models
                ns[key] = cls.deserialize_arrays(ns[key])
            elif json_like(val, 'Array'):
                ns[key] = Array.validate(val)
        return ns
            
    @root_validator
    def check_init_cond(cls, values):
        init_cond = values.get('init_cond', None)
        model = values.get('values', None)
        if model is not None:
            hists_to_init = {nm for nm,h in model.nested_histories.items()
                             if h.pad_left}
            init_hists = set(init_cond.keys())
            if hists_to_init == init_hists:
                pass
            elif hists_to_init - init_hists:
                raise ValueError("The following histories are missing initial "
                                 f"conditions: {hists_to_init - init_hists}.")
            elif init_hists - hists_to_init:
                logger.warning("Initial conditions for the following histories "
                               f"were specified but unneeded: {init_hists - hists_to_init}. "
                               "They will be ignored.")
                # raise ValueError("The following histories do not need initial "
                #                  f"conditions: {init_hists - hists_to_init}.")
                init_cond = {k: ic for k, ic in init_cond.items()
                              if k in hists_to_init}
                assert set(init_cond.keys()) == hists_to_init, \
                    "Despite being explicitely constructed to do so, the subset of initial condition keys doesn't match the histories which need initial conditions."
                values['init_cond'] = init_cond
            else:
                assert False, "There should be no code path leading here."
        # Return
        return values
        
    def dict(self, *args, exclude=None, **kwargs):
        """
        The `model` attribute is excluded by default, since it is only used
        to initialize `params`. Exporting it in each trial also wastes a lot 
        of space.
        """
        exclude = add_exclude_mask(exclude, {'model'})
        return super().dict(*args, exclude=exclude, **kwargs)
        
    # def __init__(self, params, init_cond, seed, model=None):
    #     super().__init__(params=params, init_cond=init_cond, seed=seed)

    @property
    def key(self):
        return (self.param_hash, self.ic_hash, self.seed)
    @property
    def label(self):
        # TODO: Add model ?
        return f"{self.param_hash}__{self.ic_hash}__{self.seed}"
    @property
    def param_hash(self):
        return mtb.utils.stablehexdigest(self.params.json())[:10]
    @property
    def ic_hash(self):
        ic_json = {k: Array.json_encoder(v) for k,v in self.init_cond.items()}
        return mtb.utils.stablehexdigest(str(ic_json))[:10]

    @property
    def filename(self):
        # There are no data files
        return None
    @property
    def data_path(self):
        # There are no data files
        raise NotImplementedError

    @classmethod
    def parse_label(cls, label):
        # Cannot invert hashes
        raise NotImplementedError
    @classmethod
    def parse_key(cls, key):
        # Cannot invert hashes
        raise NotImplementedError

# %%
class SyntheticDataAccessor(BaseAccessor):
    """
    Same interface as DataAccessor, but instead of retrieving data from disk,
    we simulate a model with certain parameters.

    The dataset returned by `load()` contains one DataArray for each unlocked
    history in the model.
    """
    metadata_filename = None
    Trial = SyntheticTrial

    def __init__(self,
                 sumatra_project,
                 model: Model,
                 param_sets: Sequence[Union[ModelParams, dict]]=None,
                 init_conds: Sequence[Dict[str, Array]]=None,
                 seeds: Sequence[int]=None):
        super().__init__(sumatra_project)
        self.model = model
        if param_sets is None: param_sets = []
        if init_conds is None: init_conds = []
        if seeds is None: seeds = []
        if not (len(param_sets) == len(init_conds) == len(seeds)):
            raise ValueError(
                "The following arguments must all have the same "
                f"  length:\nparam_sets (received len {len(param_sets)})\n"
                f"  init_conds (received len {len(init_conds)})\n"
                f"  seeds (received len({len(seeds)}")
        self.add_param_sets(param_sets, init_conds, seeds)

    desc_keys = BaseAccessor.desc_keys + ['model']
    def to_desc(self):
        desc_dict = super().to_desc()
        # The current data in the model is meaningless and would
        # a) add cruft, multiplying file size and b) make the output less consistent
        # The same goes for the RNG state, since that is set by the sim_keys
        model = self.model
        curtidcs = {h: h.cur_tidx for h in model.history_set}
        for h in model.history_set:
            if h._taps:
                warn("Exporting SyntheticDataAccessor while there are pending symbolic "
                     "updates is undefined.")
            h._num_tidx.set_value(h.t0idx-1)

        desc_dict['model'] = self.model.json(exclude={'rng'})

        for h, tidx in curtidcs.items():
            h._num_tidx.set_value(tidx)

        return desc_dict

    @classmethod
    def from_desc(cls, desc, sumatra_project=None):
        model_dict = pydantic.parse.load_str_bytes(
            desc['model'], content_type='json', encoding='utf-8')
        rng = shim.config.RandomStream()  # Placeholder RNG to replace the one removed in `to_desc`
        model_dict['rng'] = rng
        model_class = mtb.iotools._load_types[model_dict['ModelClass']]
        model = model_class.parse_obj(model_dict)
        ## We don't use super(), so reproduce and adapt super().from_desc
        AccessorClass = mtb.iotools._load_types[desc['AccessorClass']]
        assert AccessorClass is cls, f"{AccessorClass} is not {cls}"
        accessor = AccessorClass(sumatra_project=sumatra_project, model=model)
        if accessor.project.name != desc['project name']:
            raise AssertionError("The SyntheticDataAccessor description "
                                 "ascribes it to a different project. \n"
                                 f"Current project: {accessor.project.name}.\n"
                                 f"Project in description: {desc['project name']}")
        # xarray import with MultiIndex is kind of broken (see https://github.com/pydata/xarray/issues/4073, https://github.com/pydata/xarray/issues/1603)
        # We work around this by recreating the MultiIndex for the dict, since we know MultiIndex names
        # (among other issues, they are missing from the dict)
        index_tuples = desc['trials']['coords']['trialkey']['data']
        index = pd.MultiIndex.from_tuples(index_tuples, names=cls.Trial.keynames)
        desc['trials']['coords']['trialkey']['data'] = index
        # Call cls.Trial on each data point
        trial_dict = desc['trials']['data_vars']['trial']
        trial_dict['data'] = [cls.Trial(**{**trialdata, 'model':model}) for trialdata in trial_dict['data']]
        # Assemble trials Dataset now that dict is emended
        trials = xr.Dataset.from_dict(desc['trials'])
        accessor.add_trials(trials)

        return accessor

    def __str__(self):
        return f"Synthetic data generated with {self.model.__class__.name}"

    def add_param_sets(self,
                       param_sets: Sequence[Union[ParameterSet, dict]],
                       init_conds: Sequence[Dict[str, Array]],
                       seeds: Sequence[int]):
        """Replaces `scan_directory`."""
        if isinstance(param_sets, dict) or not isinstance(param_sets, Sequence):
            raise TypeError("`param_sets` should be a list of dictionaries or model params.")
        if not isinstance(seeds, Sequence):
            raise TypeError("`seeds` should be a list of integers.")
        if len(param_sets) != len(seeds):
            raise ValueError("Synthetic DataAccessor: `param_sets` and `seeds` "
                             "must have the same length.")
        trial_list = [self.Trial(params=Θ, init_cond=ic, model=self.model, seed=seed)
                      for Θ, ic, seed in zip(param_sets, init_conds, seeds)]
        index_tuples = [trial.key for trial in trial_list]
        index = pd.MultiIndex.from_tuples(index_tuples, names=self.Trial.keynames)
        trials = pd.DataFrame(
            [(trial.filename, trial) for trial in trial_list],
            index=index,
            columns=['filename', 'trial']
        )
        trials = trials.sort_index()
        trials.index.name="trialkey"  # Do this after sorting to avoid losing name

        self.add_trials(trials)

    def get_trial(self, trial):
        if isinstance(trial, (dict, ModelParams)):
            raise NotImplementedError(
                "The usefulness of `SyntheticDataAccessor.Trial` is unclear given that "
                "we must pass a simulation seed along with parameters. "
                "Waiting for a use-case to implement.")
            # trial = Trial(params=trial, model=self.model)
            # if trial.key not in self.trials:
            #     raise ValueError("Parameters do not match those of any defined trial.")
        elif isinstance(trial, tuple):
            # TODO: Only use this branch if format is plausibly a trial key
            trial = self.trials.sel(trialkey=trial).trial.data[()]
        else:
            assert isinstance(trial, self.Trial)
        return trial

    def load(self, trial):
        # Wraps _load. By normalizing the argument before calling the memoized
        # function, we reduce cache misses.
        return self._load(self.get_trial(trial))
        
    @lru_cache(maxsize=4)  # 4 is a wild guess
    def _load(self, trial: SyntheticTrial):
        model = self.model

        # Set model parameters to those of the trial
        model.update_params(trial.params)  # `update_params` calls `model.clear()`
        for histname, init_val in trial.init_cond.items():
            hist = model.nested_histories[histname]
            hist[:hist.t0idx] = init_val
            
        # Set the RNG for reproducible runs
        model.reseed_rngs(trial.seed)  # Raises NotImplementedError if there is more than one RNG

        # Create the data by integrating the model
        model.integrate(upto='end', histories='all')
        if hasattr(model, 'remove_degeneracies'):
            model.remove_degeneracies()

        # Construct a DataArray for each model history
        time_array = DataArray(model.time.stops_array,
                               name='time',
                               dims=['time'],
                               attrs={'dt': model.time.dt,
                                      'unit': model.time.unit})

        # Return an xarray Dataset
        unlocked_hists = {nm: h for nm, h in model.nested_histories.items()
                          if not h.locked}
        return utils.dataset_from_histories(unlocked_hists.values(),
                                            names=unlocked_hists.keys())

# %% [markdown]
# ## Example

# %%
if __name__ == "__main__":
    from sinnfull.rng import get_seedsequence, get_fit_rng, get_sim_rng
    from sinnfull.models import OUInput, TimeAxis
    from IPython.display import display
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    sim_keys = [(0,0)]  # Use only one realization
    param_keys = [(1,0)]

    time = TimeAxis(min=0, max=1, step=0.001, units=sinnfull.ureg.s)

    M = 2
    Mtilde = 2
    def generate_params(key):
        θrng = get_fit_rng(key)

        μ = θrng.normal(loc=0., scale=2., size=(M,))
        τ = θrng.lognormal(mean=0., sigma=1., size=(M,))
        σ = np.ones(shape=(M,))
        #σ = θrng.lognormal(mean=0., sigma=1., size=(M,))
        W = θrng.normal(loc=0., scale=1., size=(M,M))
        # Remove parameter degeneracy as described in models#ou-input
        Θ = OUInput.Parameters(μtilde=μ, τtilde=τ, σtilde=σ, Wtilde=W, M=M,
                               Mtilde=Mtilde)
        return OUInput.remove_degeneracies(Θ)
        # σ = np.sqrt(abs(W.sum(axis=1)))
        # μ *= σ
        # W /= σ[:,np.newaxis]**2
        # return OUInput.Parameters(μtilde=μ, τtilde=τ, σtilde=σ, Wtilde=W, M=M,
        #                           Mtilde=Mtilde)

    sim_seeds  = [get_seedsequence(key, exists_ok=True).generate_state(1) for key in sim_keys]
    param_sets = [generate_params(key) for key in param_keys]

# %%
if __name__ == "__main__":
    rng = get_sim_rng(sim_keys[0], exists_ok=True)
    model = OUInput(time=time, params=param_sets[0], rng=rng)

# %%
if __name__ == "__main__":
    data = SyntheticDataAccessor(
        sumatra_project=sinnfull.projectdir,
        model     =model,
        param_sets=param_sets,
        seeds     =sim_seeds)

# %%
if __name__ == "__main__":
    display(data.trials)

# %% [markdown]
# LRU cache works as expected on `load()` function:

# %%
if __name__ == "__main__":
    import time
    data.load.cache_clear()
    trial = data.trials.trial.data[0]
    t1=time.perf_counter(); data.load(trial); t2=time.perf_counter()
    print(f"First load:  {(t2-t1)/60*1000:.5f} ms")
    t1=time.perf_counter(); data.load(trial); t2=time.perf_counter()
    print(f"Second load:  {(t2-t1)/60*1000:.5f} ms")
    print(data.load.cache_info())

# %%
if __name__ == "__main__":
    trial = data.trials.trial.data[0]
    fig, axes = plt.subplots(1, 2, figsize=(8,3))

    X = data.load(trial)
    axes[0].plot(X.time, X.I);

    X = data.load(trial)
    axes[1].plot(X.time, X.I);

    print(data.load.cache_info())
