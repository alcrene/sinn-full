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
#     display_name: Python (sinnfull)
#     language: python
#     name: sinnfull
# ---

# %%
from __future__ import annotations

# %%
import sinnfull
if __name__ == "__main__":
    sinnfull.setup()
import logging
logger = logging.getLogger(__name__)

# %% [markdown]
# # Convergence tests
#
# Convergence testing functions are boolean functions that return *True* if an optimization problem has converged, or partially converged (e.g. latents may converge before parameters). Based on the current values in one or multiple *Recorders* instances, they return one of the values defined by `~sinnfull.optim.OptimizerStatus`. The `~sinnfull.optim.Optimizer` is expected to use this value to update its status.
#
# All convergence tests take two arguments: `recorders` and `optimizer`.
# `recorders` should be a dictionary of `Recorder` instances, with keys matching their names.
# `optimizer` should be an `Optimizer` instance.
#
# Since each convergence test requires different recorders, they need to be defined when the test function is created. This is done by passing the *name* of the recorder, which is more robust for serialization. The name is then used within the function to retrieve the optimizer::
#
#     recorder = recorders[recorder_name]
#
# All convergence tests return `NotConverged` by default, since OR-ing that value with the current status has no effect.

# %%
import math
from typing import ClassVar
from pydantic import BaseModel, conint, PositiveFloat
from sinn.utils.pydantic import initializer

# %%
from sinnfull.optim.base import OptimizerStatus, Optimizer
from sinnfull.optim.recorders import Recorder

# %%
__all__ = ["ConvergenceTest", "DivergingCost", "ConstantCost"]

# %% [markdown]
# ## Base class

# %%
# The _class_registry is a poor man's version of smttask-ml's AbstractSerializableObject
class ConvergenceTest(BaseModel):
    """
    Abstract base class for convergence tests. Maintains a registry of concrete
    subclasses to allow deserialization.
    """
    _class_registry    : ClassVar[dict] = {}
    __serialized_type__: constr(regex="^ConvergenceTest$")="ConvergenceTest"

    @classmethod
    def validate(cls, value):
        if cls is not ConvergenceTest:
            # Don't shadow the subclasses' own `validate`, which should
            # execute BaseModel's `validate`
            # Without this, the line test_type.validate(value) causes infinite recursion
            return super().validate(value)
        else:
            validation_error = (TypeError, ValueError, AssertionError)
            if isinstance(value, ConvergenceTest):
                if type(value) is ConvergenceTest:
                    raise TypeError("`ConvergenceTest` is an abstract class and "
                                    "cannot be used to instantiate objects directly.")
                else:
                    return value
            else:
                result = None
                for test_type in cls._class_registry.values():
                    try:
                        result = test_type.validate(value)
                    except validation_error:
                        pass
                    else:
                        break
                if result is None:
                    raise TypeError("Provided value does not match any of the "
                                    "known ConvergenceTest formats.\n"
                                    f"Known convergence tests: {cls._class_registry.values()}\n"
                                    f"Value received: {value}.")
                return result


# %% [markdown]
# ## `DivergingCost`
#
# Return `Failed` if the latest cost is worse than the initial cost.
#
# Recorders:
#
# - Cost recorder (e.g. `~sinnfull.optim.recorders.LogpRecorder`)

# %%
class DivergingCost(ConvergenceTest):
    """
    Parameters
    ----------
    cost_recorder:
        Recorder for a scalar cost.
        Example: `~sinnfull.optim.recorders.LogpRecorder`
    maximize:
        Whether the optimizer tries to maximize or minimize the cost value.
        True=maximize, False=minimize.
        There is no default because getting the sign of the cost wrong
        is an extremely common mistake.

    Returns
    -------
    `OptimizerStatus.Failed`
        If the current cost is less then the initial one.
    `OptimizerStatus.NotConverged`
        Otherwise
    """
    __serialized_type__: constr(regex="^DivergingCost$")="DivergingCost"
        # Used to prevent deserializing into a different subclass; each subclass must define a different name
    cost_recorder: str
    maximize     : bool

    def __call__(self, recorders: Dict[str,Recorder], optimizer: Optimizer):
        try:
            cost_recorder = recorders[self.cost_recorder]
        except KeyError:
            logger.error("Unable to apply `DivergingCost` test: no recorder "
                         f"named '{self.cost_recorder}' exists.")
            return OptimizerStatus.NotConverged
        if cost_recorder.steps[-1] != optimizer.stepi:
            # No point in performing the check if recorder hasn't been updated.
            return OptimizerStatus.NotConverged
        Δ = cost_recorder.values[-1] - cost_recorder.values[0]
            # If we are minimizing a loss, then the last value should be less
            # then the initial value => Δ should be negative
        if self.maximize:
            Δ *= -1
        if math.isnan(Δ) or (len(cost_recorder) > 10 and Δ > 0):
            logger.info("Optimization terminated because of diverging cost.")
            optimizer.outcome += ("Cost diverged.",)
            return OptimizerStatus.Failed
        else:
            return OptimizerStatus.NotConverged

ConvergenceTest._class_registry['DivergingCost'] = DivergingCost


# %% [markdown]
# ## ConstantCost
#
# Return `Converged` if the last $n$ cost values are within $ε$ of the last value.
#
# Recorders:
#
# - Cost recorder (e.g. `~sinnfull.optim.recorders.LogpRecorder`)
#
# Parameters:
#
# - $ε$: Cost tolerance. Default: 1. Default is appropriate for log-likelihood cost.
# - $n$: Number of recorded steps to consider. Default: 5. Must be at least 2.
#
# > **Remarks**
# >
# > - In order for the value of $ε$ to be independent of the cost, the employed cost should almost always be log-scaled.
# >
# > - The default `LogpRecorder` thins the recording density as the number of steps increases.

# %%
class ConstantCost(ConvergenceTest):
    """
    Parameters
    ----------
    cost_recorder: recorder name (str)
        Recorder for a scalar cost.
        Example: `~sinnfull.optim.recorders.LogpRecorder`
    tol: float > 0
        Cost values differing by a value less than `tol` are considered
        “equal” for the purpose of testing convergence.
    n: int ≥ 2
        Number of cost steps to consider, including the last one.

    Returns
    -------
    `OptimizerStatus.Converged`
        - If the last `n` cost values are within a `tol` of each other.
        - If the specified cost recorder is not attached to the optimizer.
          A message stating this is also logged at level 'Error'.
    `OptimizerStatus.NotConverged`
        Otherwise
    """
    __serialized_type__: constr(regex="^ConstantCost$")="ConstantCost"
        # Used to prevent deserializing into a different subclass; each subclass must define a different name
    cost_recorder: str
    tol          : PositiveFloat=2**-5  # ~0.03
    n            : conint(gt=1)

    def __call__(self, recorders: Dict[str,Recorder], optimizer: Optimizer):
        try:
            cost_recorder = recorders[self.cost_recorder]
        except KeyError:
            logger.error("Unable to apply `ConstantCost` test: no recorder "
                         f"named '{self.cost_recorder}' exists.")
            return OptimizerStatus.NotConverged
        if len(cost_recorder) < self.n:
            # Not enough data points yet
            return OptimizerStatus.NotConverged
        elif cost_recorder.steps[-1] != optimizer.stepi:
            # No point in performing the check if recorder hasn't been updated.
            return OptimizerStatus.NotConverged
        last_c = cost_recorder.values[-1]
        if max(abs(c-last_c) for c in cost_recorder.values[-self.n:-1]) < self.tol:
            logger.info("Optimization terminated because cost is constant.")
            optimizer.outcome += ("Cost converged.",)
            return OptimizerStatus.Converged
        else:
            return OptimizerStatus.NotConverged

ConvergenceTest._class_registry['ConstantCost'] = ConstantCost

# %%
