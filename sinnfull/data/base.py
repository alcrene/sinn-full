# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Abstract DataAccessor
#
# Defines abstract base classes for `DataAccessor` and `Trial`, which provide generic functionality which is agnostic to the data source. Concrete subclasses are defined in other modules in this directory.
#
# Classes:
# ~ `DataAccessor`
# ~ `Trial`
#
# Functions:
# ~ `get_data_format()`

# %%
from __future__ import annotations

# %%
import os
import re
import abc
import warnings
from warnings import warn
from typing import Union, List, Tuple, Optional
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import pandas as pd
import xarray as xr
from xarray import DataArray, Dataset
from pydantic import BaseModel
from pydantic.typing import ClassVar

# %%
from sumatra.projects import load_project
import mackelab_toolbox as mtb
import mackelab_toolbox.iotools

# %%
import sinnfull
from sinnfull.parameters import ParameterSet

# %% [markdown]
# TODO:
#
# - Use DataFrame instead of Dataset for trials:
#   Dataset is overkill, and support for MultiIndex (esp. wrt export) is spotty
# (https://github.com/pydata/xarray/issues/1603 and Issue #4073)
# - Remove trials.filename; use Series instead of DataFrame for trials

# %%
__all__ = ['Trial', 'DataAccessor']

# %%
data_format_re = re.compile("^data_format\.?")
def get_data_format(inputroot: str, trial: Trial):
    """
    Look for a file named 'data_format' in the directory containing `trial`;
    ``trial.data_path`` is relative to an root directory, specified by
    `input_root`.
    If no data format file is found in the trial directory, this function will
    recurse through the parents up to and including `inputroot` until it finds
    one. Thus, if your data are split across multiple directories but all share
    the same format, it suffices to place one 'data_format' file in the root
    data directory.
    """
    inputroot = Path(inputroot)
    # assert isinstance(trial, Trial)
    curpath = inputroot/trial.data_path
    data_format = None
    # This loop start in the directory containing the trial file, and work up
    # the directory tree up to and including `inputroot`.
    while curpath != inputroot:
        curpath = curpath.parent
        data_format_files = [p for p in os.listdir(curpath)
                               if data_format_re.match(p)]
        if not data_format_files:
            continue
        if len(data_format_files) > 1:
            warn("Multiple files specifying the data format in directory "
                 f"'{curpath}'. Will attempt to load in this order, and "
                 f"terminate on first success: {data_format_files}")
        for p in data_format_files:
            # TODO?: If data format gets more complex, split into its own class/dataclass
            try:
                data_format = ParameterSet(str(curpath/p))
            except ValueError:
                warn(f"Invalid data format file: {p}.")
        if data_format is not None:
            break
        else:
            continue
    return data_format

# %%
class Trial(BaseModel, abc.ABC):
    # IMPORTANT: Subclasses intended for on-disk data MUST define a `datadir` attribute

    #Force subclasses to define the 'keynames' class variable
    @property
    @abc.abstractmethod
    def keynames(self):
        pass

    class Config:
        extra = 'forbid'

    def __hash__(self):
        return hash(self.key)

    @property
    @abc.abstractmethod
    def key(self):
        raise NotImplementedError
    @property
    @abc.abstractmethod
    def label(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filename(self):
        raise NotImplementedError
    @property
    @abc.abstractmethod
    def data_path(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def parse_label(cls, label):
        raise NotImplementedError
    @classmethod
    @abc.abstractmethod
    def parse_key(cls, key):
        raise NotImplementedError


# %%
class DataAccessor(abc.ABC):

    #### Abstract methods that subclasses must defined ####

    # metadata_filename should be defined as a class attribute
    @property
    @abc.abstractmethod
    def metadata_filename(self) -> str:
        """
        Class attribute. It should take the form "pre_{subject}_post", i.e.
        it should include a placeholder named 'subject', to be replaced by
        the subject's label.
        """
        pass

    @property
    @abc.abstractmethod
    def Trial(self) -> type:
        """
        Class attribute: Trial type
        """
        pass

    @abc.abstractmethod
    def load(self, trialkey) -> xr.Dataset:
        """
        Parameters
        ----------
        trial: Trial | str (trial label)

        Returns
        -------
        xarray Dataset
        """
        pass

    ##### End of abstract methods #########################

    def __init__(self, sumatra_project=None, datadir=None):
        """
        Parameters
        ----------
        project: Path-like
            Directory containing the .smt file for the Sumatra project
        datadir: Path-like
            If is provided, that directory is scanned for data files with
            `DataAccessor.scan_directory()`.

        """
        self.project = None
        keyindex = pd.MultiIndex.from_tuples([], names=self.Trial.keynames)
        self.trials = xr.Dataset(
            data_vars={'filename': ('trialkey', np.zeros(0, dtype=object)),
                       'trial': ('trialkey', np.zeros(0, dtype=object))},
            coords={'trialkey': keyindex}
        )
        # self.trials = pd.DataFrame(columns=['filename', 'trial'])
        self.subjects = {}
        self.set_sumatra_directory(sumatra_project)
        if datadir is not None:
            self.scan_directory(datadir)

    def set_sumatra_directory(self, path=None):
        if len(self.trials.trialkey) + len(self.subjects) > 0:
            warn("Changing the Sumatra project while data is already loaded "
                 "leads to undefined behaviour.")
        if path is None:
            path = sinnfull.projectdir
        self.project = load_project(path)

    ## Convenience methods and properties ##

    @property
    def inputroot(self):
        if self.project is None:
            raise RuntimeError(f"Can't retrieve 'inputroot from {self}: "
                               "Sumatra project is not set.")
        return Path(self.project.input_datastore.root)

    def get_trial(self, trial):
        """
        Normalize `trial`: convert str, tuple, DataArray to Trial;
        effectively a wrapper for the `Trial` class.
        Returned Trial will have its :attr:`.data_path` attribute correctly set.
        This means thet `trial` must correspond to an entry in `self.trials`.
        """
        if isinstance(trial, (str, tuple, xr.DataArray)):
            trial = self.Trial(trial)
            trial = self.trials.trial.sel(trialkey=trial.key).data[()]
        else:
            assert isinstance(trial, self.Trial)
        return trial

    def get_data_format(self, trial: Trial):
        """
        Wrapper for the `get_data_format` function, which provides the project
        data directory.
        """
        return get_data_format(self.project.input_datastore.root, trial)

    ## Serialization ##

    def __init_subclass__(cls):
        # DataAccessor itself is registered manually below
        mtb.iotools.register_datatype(cls)

    desc_keys = ['project name', 'AccessorClass', 'trials']
    def to_desc(self):
        desc_dict = {
            'project name': self.project.name,  # Do something with this ? Currently just asserts in from_desc
            'AccessorClass': mtb.iotools.find_registered_typename(type(self)),
            'trials': self.trials.to_dict()
        }
        return desc_dict

    @classmethod
    def from_desc(cls, desc, sumatra_project=None, **kwargs):
        """
        **kwargs meant for subclasses, so they can pass extra parameters to __init__.
        """
        if (not isinstance(desc, dict)
            or not (set(desc.keys()) >= set(cls.desc_keys))):
            if not isinstance(desc, dict):
                info = f"Received a value of type '{type(desc)}'."
            else:
                info = f"Received a dictionary with keys {list(desc)}."
            raise TypeError(f"{cls.__name__}.from_desc expects a dictionary "
                            f"with at least the keys {cls.desc_keys}. "
                            f"{info}")
        AccessorClass = mtb.iotools._load_types[desc['AccessorClass']]
        if AccessorClass is not cls:
            # Start from the `from_desc` method of the proper class
            # If it requires the rest of this method, it will call it through
            # super(), which is why we return immediately
            return AccessorClass.from_desc(desc, sumatra_project=sumatra_project)
        accessor = AccessorClass(sumatra_project=sumatra_project, **kwargs)
        if accessor.project.name != desc['project name']:
            raise AssertionError("The DataAccessor description ascribes it to "
                                 "a different project. \n"
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
        trial_dict['data'] = [cls.Trial(**trialdata) for trialdata in trial_dict['data']]
        # Assemble trials Dataset now that dict is emended
        trials = xr.Dataset.from_dict(desc['trials'])
        accessor.add_trials(trials)
        # Assert that the trial list points to existing data files
        try:
            for trial in trials.trial.data:
                assert os.path.exists(accessor.inputroot/trial.data_path)
        except NotImplementedError:
            # Synthetic data doesn't define `data_path`
            pass
        return accessor

    # Expose serialization methods to Pydantic
    # TODO: How should we set `sumatra_project` with Pydantic ?
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        if isinstance(value, cls):
            return value
        else:
            return cls.from_desc(value)

    ## Runtime object inspection ##

    def __str__(self):
        overall_summary = f"{len(self.subjects)} subjects, " \
                          f"{len(self.trials)} total trials"
        subject_summaries = [
            f"Subject {subject.label}: {len(subject.channels)} channels, "
            f"{len(self.trials.sel(subject=subject.label))} seizure events."
            for subject in self.subjects.values()
        ]
        return overall_summary + "\n" + "\n".join(subject_summaries)

    ## Begin actual methods ##

    def scan_directory(self, datadir):
        """
        Scan the directory `datadir` for files ending with ".mat".

        TODO: More flexible condition for data files
        """
        file_list = os.listdir(self.inputroot/datadir)
        # Get the list of trials
        trial_list = [self.Trial(fn[:-4], datadir=datadir) for fn in file_list if fn[-4:] == ".mat"]
        index_tuples = [trial.key for trial in trial_list]
        index = pd.MultiIndex.from_tuples(index_tuples, names=self.Trial.keynames)
        trials = pd.DataFrame(
            [(trial.filename, trial) for trial in trial_list],
            index=index,
            columns=['filename', 'trial']
        )
        trials = trials.sort_index()
        trials.index.name = next(iter(self.trials.dims.keys()))  # Ignored by Pandas, but used by xarray
                                                                 # Will disappear on Pandas copy
        self.add_trials(trials)

    def add_trials(self, trials):
        """Add to already loaded data (`scan_directory` can be called multiple times)"""
        if isinstance(trials, xr.Dataset):
            trialdim_name = next(iter(self.trials.dims.keys()))
            trials = trials.to_dataframe()
            trials.index.name = trialdim_name  # As above, require by xarray to match dims
        with warnings.catch_warnings():
            # xarray stores the 'trials' index as an array `data`, and checks for
            # inclusion `key in data`. Since `key` is a MultiIndex, it's a tuples
            # and this raises the NumPy FutureWarning:
            # "elementwise comparison failed; returning scalar instead, but in
            # the future will perform elementwise comparison"
            # I'm not willing to patch xarray to fix this ;-P, so I just
            # silence the warning
            warnings.simplefilter('ignore', FutureWarning)
            assert all(t_idx not in self.trials.trialkey for t_idx in trials.index)
        self.trials = xr.merge((self.trials, xr.Dataset(trials)))

    def get_subject_metadata_files(self, trials) -> Generator:
        """
        .. note: The location of the subject metadata is determined by finding
        the common parent to all that subject's trials, and appending the
        class variable in `DataAcessor.metadata_filename`.
        """
        if isinstance(trials, xr.Dataset):
            trials = trials.to_dataframe()

        for label in trials.index.get_level_values('subject').unique():
            # Get all datadirs containing data for this subject
            datadirs = [trial.datadir for trial in trials.xs('NA').trial]
            # Find common root dir
            datadir = os.path.commonpath(datadirs)
            # Filename is set by class variable, which has a placeholder for 'subject'
            filename = self.metadata_filename.format(subject=label)
            path = self.inputroot/datadir/filename
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"The channel metada for subject {label} was not found at "
                    f"at the location '{path.absolute()}'.")
            yield label, path
