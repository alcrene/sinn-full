from sinn.histories import TimeAxis
from .base import *
from ._scandir import models, objectives, priors, paramsets
from .tagcolls import *

# from pathlib import Path
# basepath=Path(__file__).parent
#     # `basepath` can be used to import parameter sets with relative references:
#     # fit_hyperparams = ParameterSet(paramset_file,
#     #                                basepath=sinnfull.models.paramsets.basepath)

if 'sinnfull.models.base' not in base.PureFunction.modules:
    base.PureFunction.modules.append('sinnfull.models.base')
