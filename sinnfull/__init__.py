"""
The project-root __init__ module does things that we want to either ensure are
always done when the project package is imported, or done exactly once.

Things to ensure are always done:

- Load any module affecting dynamic types
  + pint
  + mackelab_toolbox.cgshim
- Define the following variables:
  + `projectdir`: points to the project directory
  + `labnotesdir`: points to a labnotes directory for run output, esp. plots.
    This is outside the project directory because we don't want to version
    control plots; it is defined as "projectdir/../labnotes/YYYY-MM-DD", where
    the final folder is today's date.
- Initialize logging
- Set defaults for the Sumatra records viewer


Things that must only be done once:

- Instantiate the unit registry
"""

import os
import warnings
from warnings import warn
import logging

import pint
from mackelab_toolbox.cgshim import shim, typing as mtbtyping
from mackelab_toolbox.utils import sentinel
from pathlib import Path
from types import SimpleNamespace
import smttask
import smttask.view
from smttask.view import RecordStoreView

from . import config

logging.basicConfig()
logger = logging.getLogger(__name__)

projectdir = Path(__file__).parent.parent
labnotesdir = projectdir/f"../labnotes"
reportsdir = projectdir/f"../reports"
# TODO: Move mpl settings to plotting subpackage
rcParams = SimpleNamespace(
    mpl=SimpleNamespace(
        default = projectdir/"default.mplstyle"
    )
)

# Silence the annoying SyntaxError due to NumPy using older traitlets version
# NOTE: As much as I would like to ignore::SyntaxWarning:traitlets[.*], it doesn't seem to work
# FIXME: Remove when no longer needed
# os.environ['PYTHONWARNINGS'] = "ignore::SyntaxWarning"

ureg = pint.UnitRegistry()

json_encoders = sentinel('sinnfull unset value')

# Configure Sumatra records viewer
RecordStoreView.default_project_dir = projectdir
smttask.config.load_project(projectdir)

_setup_was_run = False  # Flag to prevent running 'setup' twice
def setup(cglib: str='numpy', view_only: bool=False):
    """
    Load the specified computational graph library ('numpy' or 'theano')
    and freeze the types.
    If there are any other modules affecting the the dynamic types, make sure
    they are imported before calling `setup`.
    It is an error to call this function more than once.

    Parameters
    ----------
    cglib: 'numpy' | 'theano'

    view_only: bool
        If True, disable smttask recording.
        This is useful in notebooks meant for viewing results, to allow
        interfacing with the Sumatra recordstore without risk of modifying it.
        Otherwise, if we try to load a task which has not been run, a new
        record store entry is created.

    **Side-effects**
       - Load a CG library (through `theano_shim.load`).
       - Freeze data types (through `mackelab_toolbox.typing.freeze_types`).
       - set sinnfull.json_encoders

    Raises
    ------
    RuntimeError:
        (Through mackelab_toolbox.typing.freeze_types) if called more than once.
    """
    global _setup_was_run
    if _setup_was_run:
        if cglib == shim.config.library:
            warn(f"sinnfull.setup was already run with argument '{cglib}'. Ignoring.")
            return
        else:
            raise RuntimeError(
                "It is an error to call sinnfull.setup twice. It was previously "
                f"called with argument '{shim.config.library}'.")
    _setup_was_run = True

    ## Load shim library ##
    if cglib == 'theano':
        # When multiple tasks are run in parallel, assign a different theano
        # compile dir to each.
        # Theano compile dir cannot be changed once Theano is loaded, so
        # we use the environment variable.
        if "SMTTASK_PROCESS_NUM" in os.environ:
            THEANO_FLAGS = os.environ.get("THEANO_FLAGS", "")
            if THEANO_FLAGS:
                THEANO_FLAGS += ","
            n = os.environ['SMTTASK_PROCESS_NUM']
            base_compiledir = os.path.expanduser(f"~/.theano/smttask_process-{n}")
            THEANO_FLAGS += f"base_compiledir={base_compiledir}"
            os.environ["THEANO_FLAGS"] = THEANO_FLAGS
    shim.load(cglib)
    shim.config.floatX = 'float64'

    ## Finalize typing setup ##
    # Serialization of Pint values requires knowing the unit registry
    mtbtyping.PintUnit.ureg = ureg
    mtbtyping.safe_packages.update(['sinnfull'])
    import mackelab_toolbox.theano  # For the typing part
    import mackelab_toolbox.pymc_typing
    mackelab_toolbox.theano.freeze_theano_types()
    mtbtyping.freeze_types()

    ## Set JSON encoders ##
    # Must be done after types are cglib is loaded and types are frozen
    global json_encoders
    import sinnfull._json_encoders
    json_encoders = sinnfull._json_encoders.json_encoders

    ## Add sinnfull result types to smttask.view functions ##
    from .optim import Recorder
    smttask.view.config.data_models.extend([Recorder])

    ## Deactivate recording if `view_only` is True ##
    if view_only:
        config.view_only = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            smttask.config.record = False
