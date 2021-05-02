from ._scandir import paramsets
from .base import *
from .recorders import Recorder, DiagnosticRecorder
from .convergence_tests import ConvergenceTest, DivergingCost, ConstantCost

from .optimizers.alternated_sgd import AlternatedSGD
