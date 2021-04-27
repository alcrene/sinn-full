# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

# # DataAccessor template

# Quick Implementation overview
# -----------------------------
#
# `DataAccessor.load`  
# ~  Definiton of the data format
#
# `Trial`  
# ~ Definition of trial metadata  
# ~ Definition of trial ID data (subset of metadata)  
# ~ Conversion of file names to and from trial data
#
#
# These classes list a public API. As long as that is kept consistent, the
# customized module should work with the rest of the sinn-full workflow.
#
# The function `get_data_format` in [base.py](./base.py) walks up the data
# directory, looking for a file named 'data_format'. This may be used to store
# general metadata information. The file name searched for is determined by the
# module variable `data_format_re`.
#
# :::{admonition} Missing template values  
# :class: important  
# The template code below uses markers `>>>>>>>>>>` and `<<<<<<<<<<` to indicate sections which must be updated.
# :::

from __future__ import annotations

import os
import re
from warnings import warn
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import pandas as pd
import xarray as xr
from xarray import DataArray, Dataset
from pydantic import BaseModel
from pydantic.typing import ClassVar

from sinnfull.parameters import ParameterSet
from sinnfull.data.base import DataAccessor as BaseAccessor, Trial as BaseTrial


# ## Trial definition

class Trial(BaseTrial):
    """
    Data structure for the trial metadata.

    The main purpose of this class is parsing the trial file names, and
    converting them to and from unique keys usable to index the data set.

    The methods in this class are all class methods, meaning they can be
    executed without creating a Trial object (which is useful when you
    need their output to create a Trial object).

    Public API
    ==========

    Attributes
    ----------
    subject: str
        A string identifying the subject.
    [data attributes]: Any
        Keys storing relevant metadata for the trial.
    datadir: str
        The data directory. Value of `None` indicates unset.
    keynames: Tuple[str]  (class variable)
        Tuple of data attributes. These are assembled to form a key tuple;
        each attribute value must be hashable.

    Properties (i.e. computed attributes)
    -------------------------------------
    key: tuple
        Key to use e.g. in DataArray or DataFrame to identify the trial.
    label: str
        Human-readable label for the trial.
    filename: str
        Filename. This is only the filename, computed from the data attributes.
    data_path: Path
        Full path to the data. This requires `self.datadir` be set.

    [Additional properties may be added as needed.]

    Class methods
    -------

    parse_label (str) -> dict
        Convert a label (string) into a dict of data attributes

    parse_key (tuple) -> dict
        Convert a key (tuple) into a dict of data attributes


    Implementation requirements
    ===========================

    - Add the data attributes relevant to your data set.
      (The 'subject' does not need to be added since `BaseTrial` provides it.)
    - If these attributes are extracted from the filenames, you will need to
      add corresponding regex expressions to do so; `parse_label` should then
      use them.
      If they are extracted in another manner, this will need to be implemented.
      Note that extracting them from the data files themselves is not ideal,
      since that would require loading every data file when scanning the data
      directory.

    - Implement `key` and `label` properties.

    """
    # >>>>>> THE FOLLOWING ARE EXAMPLE TRIAL ATTRIBUTES; ADAPT AS REQUIRED >>>>>
    subject:str
    date   :datetime.date
    session:int

    # >>>>>>>> datadir & keynames
    # > `datadir` MUST be defined for on-disk data sets.
    # > `keynames` defines which attributes uniquely define a Trial.
    datadir :str
    keynames:ClassVar = ('subject', 'date', 'session')
        # >>>>>> Change key names to the trial attributes defined above.

    # The following defaults are used ONLY when one of the ALTERNATIVE Trial
    # initializations is used; during normal keyword-based initialization, the
    # attributes remain REQUIRED
    default_kwargs = {"datadir": ""}

    # These regexes are used to parse file names; they invert __str__.
    # The pattern for subject is just an example
    # >>>>>>>>> Add patterns associated to data keys
    _re_patterns = dict(
        subject     = re.compile("^[a-zA-Z0-9]*(?=_)"),     # Everything before first '_'
    )

    def __init__(self, label=None, **kwargs):
        """
        There should be no reason to specify both an trial label and kwargs,
        but if this happens, kwargs take precedence.
        """
        if label is not None:
            # Alternative constructors, using Trial(val)
            # val is a string: treat it as a label
            # val is a tuple: treat as a key
            # val is a DataArray: treat as a coordinate value, as returned by self.trials[trialkey].trial
            if isinstance(label, str):
                default_kwargs = {**self.default_kwargs, **self.parse_label(label)}
            elif isinstance(label, tuple):
                default_kwargs = {**self.default_kwargs, **self.parse_key(label)}
            else:
                assert isinstance(label, DataArray)
                assert label.ndim == 0
                assert list(label.coords.keys()) == ['trial']
                default_kwargs = {**self.default_kwargs, **self.parse_key(label.data[()])}
            kwargs = {**default_kwargs, **kwargs}
        super().__init__(**kwargs)

    def __hash__(self):
        return hash(self.key)

    # >>>>> MISSING >>>>
    @property
    def key(self):
        raise NotImplementedError
        # return (self.attr_a, attr_b, ...)
    @property
    def label(self):
        raise NotImplementedError
        # s = f"a{self.attr_a}__b{self.attr_b}__"
        # return s
    # <<<<<<<<<<<<<<<<<<

    @property
    def filename(self):
        return f"{self.label}.mat"  # <<<<< CHANGE EXTENSION AS REQUIRED
    @property
    def data_path(self):
        """Path to the trial data, relative to datastore root."""
        return Path(self.datadir)/self.filename

    @classmethod
    def parse_label(cls, label: str) -> dict:
        """Convert a label (string) into a dict of data attributes"""
        kws = {kw: pat.search(label)
               for kw, pat in cls._re_patterns.items()}
        # Ensure we only index successful matches (unsucessful ones return None)
        kws = {kw: v[0] if v else v for kw, v in kws.items()}
        return kws
    @classmethod
    def parse_key(cls, key: tuple) -> dict:
        """Convert a key (tuple) into a dict of data attributes"""
        return {keyname: keyval for keyname, keyval in zip(cls.keynames, key)}

# ## DataAccessor definition

class DataAccessor(DataAccessorBase):
    """
    Data are expected to be spread across multiple files, each file corresponding
    to a trial.

    Public API
    ==========

    Initialization
    --------------
        sumatra_project: Path-like | None (default=None)
        datadir        : Path-like | None (default=None)

    Attributes
    ----------
    project: `sumatra.projects.Project` instance
        The Sumatra project used to track simulation and analyses.
        In the context of the DataAccessor, used to determine the location
        of data directories following the `smttask` model.
    trials: xarray.Dataset
        A summary of available data files; one use for this is to choose a trial
        when sampling across trials, and then to determine the file path from
        which to load that trial.
        The `trials` Dataset is filled by calling `DataAccessor.scan_directory`;
        if `datadir` is provided during initialization, that directory is
        scanned for data files.
    load: (trial) -> xarray.Dataset
        Loads the dataset saved at the location corresponding to `trial`.

    Implementations of DataAccessor may add any number of additional attributes
    to store additional metadata.
    An example of this would be equipment specifications which are are relevant
    to the analysis, but not included in the data files.
    However, the data themselves should generally not be stored in this object,
    but rather loaded dynamically with `~DataAccessor.load`, to avoid
    unnecessary strain on the system memory.

    Methods
    -------
    In the argument definitions below, “trial descriptor” corresponds to any
    value accepted by `Trial`'s initializer.

    - `load`(trial descriptor) -> `xarray.Dataset`
      Load data file associated to trial.
    - `Trial`(trial descriptor) -> `Trial` object
      Wrapper for the bare `Trial` initializer, which associates it with the
      path to the data directory.
    - `to_desc` () -> dict
      Return a dictionary containing a complete description of the Accessor.
      The default implementation returns a dict with three keys:
      'project name', 'AccessorClass' and 'trials'.
    - `from_desc` (dict) -> DataAccessor
      Inverse function to `to_desc`. Reads the 'AccessorClass' entry to
      determine which DataAccessor class to create.

    Implemementation requirements
    =============================
    The following **must** be defined by the subclass (otherwise a
    TypeError is raised during instantiation)

    Trial
       (Class attribute: Type) The type used to store trial info.

    metadata_filename
       (Class attribute: str) Metada files store subject-specific, like channel
       locations.

    load
       (Method) Adapt to the data storage format. This is where the specific of
       the data format are encoded, and where most of the implementation work
       will be.

    One may also want to specialize the following methods, although default
    implementations are provided by DataAccessorBase.

    __str__
       One could add relevant additional metadata.

    scan_directory
       One could need to adapt to a different layout of data files.

    to_desc / from_desc
       If `trials` is not stored as a Dataset, or is not sufficient to
       reconstruct the Accessor, these methods may have to be specialized.

    """
    # Should include a placeholder named 'subject', to be replaced by
    # the subject's label.
    metadata_filename = "great_dataset_{subject}_metadata.placeholder"
    Trial = Trial

    def load(self, trial):
        """
        Parameters
        ----------
        trial: Trial | str (trial label)

        Returns
        -------
        xarray Dataset

        Todo
        ----
        Load multiple trials
        """
        # Remove when subclassing
        raise NotImplementedError

        # Normalize `trial`: if string, convert appropriately
        # TODO: Not sure if accepting the tuple, DataArray initializers for Trial is the best idea
        trial = self.get_trial(trial)
        inputroot = Path(self.project.input_datastore.root)

        # Load data
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # DO ALL THE DATA FILE PARSING STUFF HERE
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # Return Dataset
        # (example assignments; eeg and onset_windows would be DataArray objects)
        dataset = Dataset({'eeg': eeg,
                           'onset_windows': onset_windows})
        dataset.time.attrs['dt'] = dt
        dataset.time.attrs['unit'] = time_unit

        return dataset
