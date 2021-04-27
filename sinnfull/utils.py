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
#   kernelspec:
#     display_name: Python (comp)
#     language: python
#     name: comp
# ---

# %% [markdown]
# # Miscellaneous utility functions

# %%
"""
Manifest
========

Interactive use / output management
-----------------------------------

get_figdir(location: str, date=None) -> Path:
   Return a computed location from a set of standard locations for figure output.

Packaging utilities (aka hacks)
-------------------------------
HideLocalDirForImports (context manager)
    Prevent parameters.py and typing.py from shadowing packages of the same name.

Notebook utilities
------------------

add_to(clsname: str)
   Decorator used to attach a method to a class after it has been defined.

add_property_to(clsname: str)
   Decorator used to attach a property to a class after it has been defined.

in_ipython() -> bool:
    Check whether we're in an ipython environment, including jupyter notebooks.

Record & data set utilities
---------------------------

dataset_from_histories(histories: Iterable) -> Dataset:
    Combine the data from multiply histories into an `~xarray.Dataset`.

recursive_dict_update(orig: dict, update_dict: dict, allow_new_keys: bool=False)
    Update the values of `orig` with those of `update_dict`.
    Updates are performed in place.

recursive_set_value(orig: dict, update_dict: dict, allow_new_keys: bool=False):
    Same behaviour as `recursive_dict_update`, but for values in `orig` which
    are shared variables, use `set_value` instead of assignment.

get_field_values(records: Sequence[NamedTuple]) -> Dict[str,list]:
    Find all unique field values from a list of named tuples.
    Returns {field name: field values}.

Model definition and execution
------------------------------

draw_model_sample(model, key, n=1)
    Draw `n` samples from the model.
    Wrapper around `pymc3.sample_prior_predictive` which ensures it is
    reproducible.

Task creation
-------------
generate_task_from_nb(input_path: str, *, parameters: ParameterSet, exec_environment: str, return_val: str, ...) -> Task | taskdesc (JSON) | notebook | None
    Execute a parameterized notebook which creates and saves an smttask
    task description file.

papermill_parameter_block(parameters: ParameterSet):
    Print the parameters as they would be passed to papermill, in a block
    that can be pasted into the target notebook.

"""

# %%
from __future__ import annotations

# %%
from warnings import warn
from collections.abc import Iterable
import xarray as xr
import theano_shim as shim
import mackelab_toolbox as mtb
import mackelab_toolbox.utils

# %% [markdown]
# DEVNOTE: Imports specific to a section may be placed at the top of that
# section, so that they are easier to identify if it is spun out into its own
# module.
#
# DEVNOTE2: No module-level imports of sinnfull modules. It must be possible to
# import 'sinnfull.utils' without importing the rest of sinnfull. If necessary,
# place an import within the function requiring it (although, if the function
# depends on another sinnfull module, it maybe shouldn't be in this 'utils' module).


# %% [markdown]
# ---------------
# ## Constants

# %% [markdown]
# NOT_SET = mtb.utils.sentinel("value not set")

# %% [markdown]
# ---------------
# ## For interactive use / output management

# %%
import os.path
from pathlib import Path
import tempfile
from datetime import datetime

# %%
def get_figdir(location: str, date=None) -> Path:
    """
    Return a computed location from a set of standard locations for figure output.
    Also makes sure the returned directory exists.
    :param:location: One of:
        'labnotes': A subdirectory of the labnotes directory giving the date.
        'reports': A subdirectory of the reports directory giving the date.
        'tmp': A standard temporary directory.
    :param:date: A parameter which datetime.datetime can parse.
        If a string, will be parsed with ``datetime.strptime(date, "%Y-%m-%d)``.
        If no argument is given, and the location expects a date,
        the current date is used.
    """
    import sinnfull  # Avoid import cycles
    if date is None:
        date = datetime.today()
    elif isinstance(date, datetime):
        pass
    elif isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    else:
        date = datetime(date)
    locations = {
        'labnotes': sinnfull.labnotesdir,
        'reports': sinnfull.reportsdir,
        'tmp': tempfile.TemporaryDirectory()
    }
    path = locations[location]
    if location in ['labnotes', 'reports']:
        path /= f"{date.strftime('%Y-%m-%d')}"
    if not os.path.exists(path):
        path.mkdir(exis_ok=True)
    return path

# %% [markdown]
# ---------------
# ## Notebook utilities

# %%
def add_to(clsname: str):
    """
    Decorator. Use to attach a method to a class after it has been defined.
    Has an effect only in the module run as the __main__ script – this avoids
    obscuring tracebacks when the module is imported.

    .. rubric:: Rational
       The purpose of this decorator is to allow a more literate form of coding
       within Jupyter notebooks, by splitting a class definition across multiple
       cells separated by Markdown-formatted explanations. This is especially
       relevant for notebooks defining algorithms or data-driven code.
       Such notebooks have dual use: on their own, they serve as documentation,
       and we are more liberal with using necessary namespace hacks to achieve
       this. However, these notebooks are also the implementations themselves
       and used as importable modules. In this case namespace hacks are more
       problematic, because the prevent Python tracebacks from giving sensible
       line numbers when debugging.

    .. rubric:: Usage requirements
       - This decorator is designed to work with notebook paired to a plain
         Python file with Jupytext.
         In the paired Python file, all markdown cells are converted to comments
         and become transparent to the interpreter.
       - Class definition cells may be separated by markdown, but not by other
         code cells. In the .py file, code cells are effectively merged together.
       - Class definition cells must be properly indented, so that subsequent
         classes become part of the class definition when cells are merged.

    .. note::
       The class name is passed as a string, not as a class instance.

    Usage
    ----------

    #%%
    class Foo:
        pass

    #%%
    Markdown text

    #%%
        @add_to('Foo')
        def bar(self):
            return 1
    """
    def attach_method(f):
        if isinstance(f, (staticmethod, classmethod)):
            globalvars = f.__func__.__globals__
            funcname = f.__func__.__name__
            qualname = f.__func__.__qualname__
            module = f.__func__.__module__
        else:
            globalvars = f.__globals__
            funcname = f.__name__
            qualname = f.__qualname__
            module = f.__module__
            # Equivalent to globals() in the namespace containing f
        try:
            # clsname should only be present in globalvars if the source
            # was executed in a notebook, with definition split across cells.
            cls = globalvars[clsname]
            setattr(cls, funcname, f)
            return getattr(cls, funcname)
        except KeyError:
            # Paths that can lead here:
            # - imported module into a script (normal path I)
            #   __name__ == whatever, IPython = False
            #   Nothing to do; assume we are still within the class, and the
            #   method will be attached normally.
            # - imported module into another IPython notebook (normal path II)
            #   __name__ == whatever, IPython = True
            #   Nothing to do; assume we are still within the class.
            # - Module executed on its own
            #   __name__ == "__main__", IPython = False
            #   Nothing to do; assume we are still within the class.
            # - Executing as __main__ within an IPython notebook
            #   __name__ == "__main__", IPython = True
            #   This is an error - the class should already have been defined.
            if in_ipython() and module == "__main__":
                raise RuntimeError(
                    f"sinnfull.utils.add_to: Cannot attach {qualname} to "
                    f"class {clsname} because it is not (yet?) defined. Could "
                    "it be that you misspelled the class name ?")
            return f
    return attach_method

# %%
def add_property_to(clsname):
    # See comments in `add_to`
    def attach_property(f):
        globalvars = f.__globals__
        try:
            cls = globalvars[clsname]
            setattr(cls, f.__name__, property(f))
            return getattr(cls, f.__name__)
        except KeyError:
            if in_ipython() and f.__module__ == "__main__":
                raise RuntimeError(
                    f"Cannot attach {f.__qualname__} to class {clsname} "
                    "because it is not (yet?) defined. Could it be that you "
                    "misspelled the class name ?")
            return property(f)
    return attach_property

# %%
def in_ipython() -> bool:
    """
    Check whether we're in an ipython environment, including jupyter notebooks.
    Copied from `pydantic.utils.in_ipython`
    """
    try:
        eval('__IPYTHON__')
    except NameError:
        return False
    else:
        return True


# %% [markdown]
# ---------------
# ## Record & data set utilities

# %%
from typing import Iterable, Sequence, NamedTuple, Dict
import xarray as xr

# %%
def dataset_from_histories(histories: Iterable) -> xr.Dataset:
    # TODO: How should we deal with padding ?
    data_arrays = {}
    time_array = None
    for h in histories:
        # Use the same time array for all history DataArrays
        if time_array is None:
            time_array = xr.DataArray(h.time.stops_array,
                                      name='time',
                                      dims=['time'],
                                      attrs={'dt': h.time.dt,
                                             'unit': h.time.unit})
        else:
            assert (h.time.stops_array == time_array.data).all()
            assert h.time.dt == time_array.attrs['dt']
            assert h.time.unit == time_array.unit
        hist_dims = [f"{h.name}_dim{i}" for i in range(h.ndim)]
        array = xr.DataArray(
            h.get_trace(),
            name=h.name,
            coords={'time': time_array},
            dims=['time'] + hist_dims,
            attrs={'dt': time_array.attrs['dt']}
        )
        data_arrays[h.name] = array

    # Return an xarray Dataset
    return xr.Dataset(data_arrays)

# %%
def recursive_dict_update(orig: dict, update_dict: dict, allow_new_keys: bool=False):
    """
    Update the values of `orig` with those of `update_dic`.
    If neither dict is nested, equivalent to ``orig.update(update_dict)``.
    If `update_dict` contains dictionaries with the same key as in `orig`,
    they are updated recursively (i.e. values in `orig` but not in `update_dict`
    are left untounched.
    Returns `None`: Updates are done in place.
    """
    for k, v in update_dict.items():
        if k in orig and isinstance(v, dict) and isinstance(orig[k], dict):
            recursive_dict_update(orig[k], v, allow_new_keys)
        elif k in orig or allow_new_keys:
            orig[k] = v
        else:
            raise KeyError(f"Key '{k}' is not a key in the dictionary "
                           "being updated.")

# %%
def recursive_set_value(orig: dict, update_dict: dict, allow_new_keys: bool=False):
    """
    Update the values of `orig` with those of `update_dic`.
    Same behaviour as `recursive_dict_update`, but for values in `orig` which
    are shared variables, use `set_value` instead of assignment.
    Returns `None`: Updates are done in place.
    """
    for k, v in update_dict.items():
        if k in orig and isinstance(v, dict) and isinstance(orig[k], dict):
            recursive_set_value(orig[k], v, allow_new_keys)
        elif k in orig and shim.isshared(orig[k]):
            if shim.is_symbolic(v):
                raise TypeError("It is an error to call `set_value()` with "
                                "a symbolic variable. Most likely the values "
                                "in `update_dict` should be non-symbolic. ")
            orig[k].set_value(v)
        elif k in orig or allow_new_keys:
            orig[k] = v
        else:
            raise KeyError(f"Key '{k}' is not a key in the dictionary "
                           "being updated.")

# %%
def get_field_values(records: Sequence[NamedTuple]) -> Dict[str,list]:
    """
    Find all unique field values from a list of named tuples.
    Returns {field name: field values}.
    """
    fields = records[0]._fields
    return {field: sorted(set(key[i] for key in records))
            for i, field in enumerate(fields)}

# %% [markdown]
# ---------------
# ## Task creation

# %%
# Used in: run
import os
import tempfile
from pathlib import Path
import smttask
from mackelab_toolbox.pymc_typing import PyMC_Model
from sinnfull.models.objective_types import ObjectiveFunction

# %%
import json
from pydantic import BaseModel
from sinnfull import json_encoders

# %%
default_output_nb = os.path.join(tempfile.gettempdir(), "out.ipynb")
default_task_save_location = os.path.join(tempfile.gettempdir(), "sinnfull_tmp_task")
# To get around papermill attempting to convert all arguments to JSON,
# these types are serialized before calling `papermill.execute_notebook()`
class DummyModel(BaseModel):
    class Config:
        json_encoders = json_encoders
json_encoder_fn = DummyModel.__json_encoder__
# json_encoders = {
#     ObjectiveFunction: lambda objective_function: objective_function.json(),
#     PyMC_Model: PyMC_Model.json_encoder
# }

# %%
def generate_task_from_nb(
    input_path,
    output_path=default_output_nb,
    parameters=None,
    engine_name=None,
    request_save_on_cell_execute=False,
    prepare_only=False,
    kernel_name=None,
    progress_bar=False,
    log_output=False,
    stdout_file=None,
    stderr_file=None,
    start_timeout=60,
    report_model=False,
    cwd=None,
    exec_environment='papermill',
    create_task_directory=True,
    return_val='task',
    **engine_kwargs
    ):
    """
    Execute a parameterized notebook which creates and saves an smttask
    task description file.

    Implemented as a simple wrapper around `papermill.execute_notebook` which
    changes some defaults so that they make more sense for the intended purpose:

    - Prevents most saves of the notebook, since we are only executing it for
      side-effects. (Changes default to not save after every cell.)
    - For any left-over saves, sets a default output path in the system's
      temporary directory.
    - Adds the 'exec_environment' parameter, and sets it to 'papermill' by default.
    - Sets the CWD to the current directory by default.

    .. Important:: The 'exec_environment' must be listed in the notebook's
       parameter cell, otherwise it is ignored. It should be used as appropriate
       within the notebook, e.g. to skip the execution of unneeded cells.

    .. Note:: At present, papermill does not seem to support non-standard
       parameter types, such as pint dimensioned quantities. Essentially the
       writeout calls json.dumps with the default arguments. This is part of
       the reason for disabling writing.

    .. Note::
       If `parameters` does not define `task_save_location`, it is added to
       `parameters` and defaults to a location in the system's temporary folder.

    .. Hint::
       If the execution of the notebook fails, use `papermill_parameter_block`
       to reproduce the problematic parameters within the notebook.

    Parameters
    ----------
    Most parameters as for `papermill.execute_notebook`.

    exec_environment: 'papermill' (default) | 'notebook' | 'module'
        Provided for completeness. Default value should always be appropriate.

    create_task_directory: bool
        If the directory given by `parameters.task_save_location`, doesn't
        exist, the default is to use that as a _file_ name for the task. If
        multiple tasks are created with the same name, they then overwrite
        each other. To mitigate this common mistake, by default a heuristic
        is used to check if the provided save location was intended as a
        directory, and if it is missing, it is created.
        Passing `False` turns off this heuristic.
        True: IF `parameters.task_save_location` doesn't exist, AND it doesn't
              contain the strings '.json' or '.taskdesc', THEN create a
              directory at that location.
              OTHERWISE do nothing.
        False: Never create a directory.

    return_val: 'task' (default) | 'taskdesc' | 'notebook' | 'none'
        What the function should return.
        'task': (default) Parse the task description and return a Task object.
        'taskdesc': Return the unparsed task description (JSON string).
            Avoids the computation of parsing a task.
        'notebook': Return the result of `papermill.execute_notebook`
            Can be alternatively specified as 'nb'.
        'none': Return nothing.


    """
    import papermill  # Import here to avoid required always requiring papermill dependency

    parameters['exec_environment'] = exec_environment
    task_save_location = parameters.get('task_save_location', default_task_save_location)
    parameters['task_save_location'] = str(task_save_location)
    if cwd is None:
        cwd = os.getcwd()
    if (create_task_directory and not os.path.exists(task_save_location)
        and '.json' not in task_save_location
        and '.taskdesc' not in task_save_location):
        os.mkdir(task_save_location)

    # Sanitize parameters
    for k, v in parameters.items():
        if isinstance(v, BaseModel):
            parameters[k] = v.json(encoder=json_encoder_fn)
        else:
            for T, encoder in json_encoders.items():
                if isinstance(v, T):
                    parameters[k] = json.dumps(v, default=json_encoder_fn)
                    break

    # Monkey patch to prevent papermill from saving notebook
    old_write_ipynb = papermill.execute.write_ipynb
    papermill.execute.write_ipynb = lambda nb, path: None
    # Execute notebook
    nb = papermill.execute_notebook(
        str(input_path), output_path=str(output_path),  # str() converts Path to plain str
        parameters=parameters,
        engine_name=engine_name, request_save_on_cell_execute=request_save_on_cell_execute,
        prepare_only=prepare_only,
        kernel_name=kernel_name, progress_bar=progress_bar, log_output=log_output,
        stdout_file=stdout_file, stderr_file=stderr_file, start_timeout=start_timeout,
        cwd=cwd, **engine_kwargs
        )
    # Remove monkey path
    papermill.execute.write_ipynb = old_write_ipynb

    if os.path.isdir(task_save_location):
        # If task_path is a directory, taskdesc was placed inside and we don't
        # know how it was named.
        # TODO? Have a cell in the template output the save location, so we
        # can extract it from `nb` ?
        task_path = None
    else:
        task_path = Path(task_save_location).with_suffix(".taskdesc.json")
    if return_val == 'task':
        if task_path is None:
            warn("Unable to determine task output path – did you specify a "
                 "directory as the task save location ?")
            return
        return smttask.Task.from_desc(task_path)
    elif return_val == 'taskdesc':
        if task_path is None:
            warn("Unable to determine task output path – did you specify a "
                 "directory as the task save location ?")
            return
        with open(task_path, 'r') as f:
            taskdesc = f.read()
        return taskdesc
    elif return_val in ['notebook', 'nb']:
        return nb
    else:
        if return_val != 'none':
            warn(f"Unrecognized value '{return_val}' for `return_val`.")
        return

# %%
def papermill_parameter_block(parameters: ParameterSet):
    """
    Print the parameters as they would be passed to papermill, in a block
    that can be pasted into the target notebook.
    This is useful when debugging a call to `generate_task_from_nb`, to set
    the notebook with the same parameters as those that cause the error.

    Parameters
    ----------
    parameters: ParameterSet
        Arguments which would be passed as “parameters” to
        `generate_task_from_nb`.

    Returns
    -------
    None  (value is printed)
    """
    ps_str = {}
    # Sanitize parameters
    for k, v in parameters.items():
        if isinstance(v, BaseModel):
            ps_str[k] = "'''"+v.json(encoder=json_encoder_fn)+"'''"
        else:
            for T, encoder in json_encoders.items():
                if isinstance(v, T):
                    ps_str[k] = "'''"+json.dumps(v, default=json_encoder_fn)+"'''"
                    break
            else:
                ps_str[k] = repr(v)
        # HACK: To allow copy-paste, we replace '\n' by '\\n'. Other escape sequence may still cause problems
        ps_str[k] = ps_str[k].replace('\n','\\n').replace('\\','\\\\')
    # Print parameters
    print("\n".join(f"{k} = {v_str}" for k,v_str in ps_str.items()))

# %%
script_args = {}
def run_as_script(module_name: str, package: str=None, **parameters):
    """
    Import (or reload) a module, effectively executing it as a script.
    The imported module can retrieve parameters, which are stored in
    `sinnfull.utils.script_args`.

    Alternative to generate_task_from_nb, which doesn't use papermill.
    Instead a global variable ("script_args") is used to pass parameters.
    The use of a the global variable `script_args` to pass parameters is
    definitely a hack, but probably no worse than the magic papermill does.
    The main advantage is that we avoid papermill's overhead
    (~2 s per execution) and the need to parse parameters.

    Automatically adds the parameter `exec_environment` and sets it to
    'script' if it is not provided.

    Parameters
    ----------
    module_name: Name of the module as it appears in sys.modules
    package: If the module has not yet been imported, this is passed
        to `importlib.import_module`.
        It is required when `module_name` is relative.
    **parameters: Parameters to pass to the script
    """
    global script_args
    import importlib
    import sys
    parameters = parameters.copy()
    if 'exec_environment' not in parameters:
        parameters['exec_environment'] = 'script'
    script_args[module_name] = parameters
    if module_name in sys.modules:
        importlib.reload(module_name)
    else:
        importlib.import_module(module_name, package=package)
