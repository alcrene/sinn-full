from . import config
from .typing_ import *
from .config import pretty_names, BokehOpts
from .record_store_viewer import (
    RSView, FitData, FitCollection, ColorEvolCurvesByMaxL,
    ColorEvolCurvesByInitKey)

import seaborn as sns

# Set plotting defaults
sns.set(rc=config.rcParams)
