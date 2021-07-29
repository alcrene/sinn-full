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
# Whenever a test returns, it updates the value of its *last_result* attribute. By default, *last_result* is set to `NotConverged`.
#
# All convergence tests define a *skip* condition (by default, this condition is to skip of the current optimizer step is different from the current recorder step). When a test is *skipped*, it returns the value stored as *last_result*.

# %%
import math
from typing import ClassVar, List, Dict
from pydantic import BaseModel, PrivateAttr, conint, confloat, PositiveFloat
from sinn.utils.pydantic import initializer

# %%
from sinnfull.optim.base import OptimizerStatus, Optimizer
from sinnfull.optim.recorders import Recorder

# %%
if __name__ == "__main__":
    import numpy as np
    import holoviews as hv
    hv.extension('bokeh')

# %%
__all__ = ["ConvergenceTest", "DivergingCost", "ConstantCost", "ConstantParams"]

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
    last_result       : OptimizerStatus = OptimizerStatus.NotConverged

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
    
    # DEVNOTE: If we find the need for __or__ see the note in CombinedConvergenceTest
    def __and__(self, other):
        return CombinedConvergenceTest(test1=self, test2=other)
            
    def skip(self, recorders: List[Recorder], optimizer: Optimizer) -> bool:
        """
        Normally called at the top of a __call__ method: implements basic checks
        to see if we should skip the test for this optimization step.
        Note that in contrast to __call__, this method expects a list of Recorders
        """
        return any(rec.steps[-1] != optimizer.stepi for rec in recorders)
            # No point in performing the check if recorder hasn't been updated.


# %%
class CombinedConvergenceTest(ConvergenceTest):
    """
    Class used as the result of a bitwise AND between two tests;
    in normal usage it should not need to be instantiated directly.
    Note that the combination is not symmetric: if the first test fails, or
    returns 'NotConverged', the second test is never executed.
    """
    __serialized_type__: constr(regex="^CombinedConvergenceTest$")="CombinedConvergenceTest"
        # Used to prevent deserializing into a different subclass; each subclass must define a different name
    test1: ConvergenceTest
    test2: ConvergenceTest
        
    def __call__(self, recorders: Dict[str,Recorder], optimizer: Optimizer):
        result1 = self.test1(recorders, optimizer)
        # DEVNOTE: To also support bitwise OR, we would need to abort on
        #          'Converged' rather than 'NotConverged'
        if result1 is OptimizerStatus.NotConverged or result1 is OptimizerStatus.Failed:
            return result1
        else:
            return result1 & self.test2(recorders, optimizer)

ConvergenceTest._class_registry['CombinedConvergenceTest'] = CombinedConvergenceTest


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
    Detects a failing fit; a failure is reported if at least one of these
    conditions is met:

    - The latest loss is either `NaN` or infinity.
    - The loss is worse than it was when the fit started.
      (Whether "worse" means greater or lesser than is determined by the flag
      `maximize`.)

    Since the evolution of the loss is generally not monotone, some tolerance
    is applied for the second condition.

    Parameters
    ----------
    cost_recorder:
        Recorder for a scalar loss.
        Example: `~sinnfull.optim.recorders.LogpRecorder`
    maximize:
        Whether the optimizer tries to maximize or minimize the cost value.
        True=maximize, False=minimize.
        There is no default because getting the sign of the cost wrong
        is an extremely common mistake.
    tol:
        Tolerate the cost going below its initial value by this amount.
        This is especially important when starting fits from already reasonable
        values, since the progress is stochastic and even a good fit may dip
        below the initial value.

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
    tol          : confloat(gt=0.)=100.

    def __call__(self, recorders: Dict[str,Recorder], optimizer: Optimizer):
        try:
            cost_recorder = recorders[self.cost_recorder]
        except KeyError:
            logger.error("Unable to apply `DivergingCost` test: no recorder "
                         f"named '{self.cost_recorder}' exists.")
            return OptimizerStatus.NotConverged
        #if cost_recorder.steps[-1] != optimizer.stepi:
        #    # No point in performing the check if recorder hasn't been updated.
        #    return OptimizerStatus.NotConverged
        if self.skip([cost_recorder], optimizer):
            return self.last_result

        Δ = cost_recorder.values[-1] - cost_recorder.values[0]
            # If we are minimizing a loss, then the last value should be less
            # then the initial value => Δ should be negative
        if self.maximize:
            Δ *= -1
        if not math.isfinite(Δ) or (len(cost_recorder) > 10 and Δ > self.tol):
            msg = "Optimization terminated because of diverging loss."
            if math.isnan(Δ):
                msg += " (Loss is NaN)."
            elif math.isinf(Δ):
                msg += " (Loss is infinite)."
            else:
                msg += f" (Loss is {cost_recorder.values[-1]}; it was {cost_recorder.values[0]} at the start of the fit.)"
            logger.info(msg)
            optimizer.outcome += ("Cost diverged.",)
            self.last_result = OptimizerStatus.Failed
        else:
            self.last_result = OptimizerStatus.NotConverged
        return self.last_result

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
# :::{admonition} Remarks
#
# - In order for the value of $ε$ to be independent of the cost, the employed cost should be log-scaled.
#
# - The default `LogpRecorder` thins the recording density as the number of steps increases.
#
# - It is not uncommon for the cost to be constant or near-constant, but for the parameters to still evolve. Thus it is recommended to pair this test with `ConstantParams`; placing `ConstantParams` first ensures that the more expensive `ConstantParams` test is computed only when necessary:
#   ```python
#   constant_params = ConstantCost(...) & ConstantParams(...)
#   ```
#   The result of both tests are AND-ed together, so only bits which are 1 in both remain 1.
#   (Exception: If either test returns `Failed`, the result is always `Failed`.)
#   
# - Note that `ConstantCost` returns the optimizer status `Converged` on success. To avoid setting the convergence state for both parameters and latents, it can be combined with the bitwise AND. For example:
#   ```python
#   constant_cost() & OptimizerStatus.ParamsConverged
#   ```
#   This is not necessary if the test is combined with another test, as above.
# :::

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
    tol          : PositiveFloat=2**-7  # ~0.0078
    n            : conint(gt=1)

    def skip(self, recorders: List[Recorder], optimizer: Optimizer) -> bool:
        return any(len(rec) < self.n for rec in recorders) or super().skip(recorders, optimizer)
            # First condition: Not enough data points yet
        
    def __call__(self, recorders: Dict[str,Recorder], optimizer: Optimizer):
        try:
            cost_recorder = recorders[self.cost_recorder]
        except KeyError:
            logger.error("Unable to apply `ConstantCost` test: no recorder "
                         f"named '{self.cost_recorder}' exists.")
            return OptimizerStatus.NotConverged
        #if len(cost_recorder) < self.n:
        #    # Not enough data points yet
        #    return OptimizerStatus.NotConverged
        #elif cost_recorder.steps[-1] != optimizer.stepi:
        #    # No point in performing the check if recorder hasn't been updated.
        #    return OptimizerStatus.NotConverged
        if self.skip([cost_recorder], optimizer):
            return self.last_result

        last_c = cost_recorder.values[-1]
        if max(abs(c-last_c) for c in cost_recorder.values[-self.n:-1]) < self.tol:
            if "Cost converged." not in optimizer.outcome:
                logger.info("Optimization terminated because cost is constant.")
                optimizer.outcome += ("Cost converged.",)
            self.last_result = OptimizerStatus.Converged
        else:
            self.last_result = OptimizerStatus.NotConverged
        return self.last_result

ConvergenceTest._class_registry['ConstantCost'] = ConstantCost

# %% [markdown]
# ## Constant params
#
#

# %% [markdown]
# The `ConstantParams` test applies the following heuristic condition:
# - Compute a linear regression for the parameter evolution from its most recent values.
# - Using the regression's slope, project the current parameter value for *twice the number of current time steps*.
# - If the *relative difference* of projected value is less than some set threshold, declare the parameter converged.
#
# Doubling the number of time steps accounts for the fact that fit dynamics tend to slow down as they approach the fixed point. Using the relative difference allows the definition of a threshold that is at least plausibly insensitive to the parameter's scale.
#
# Denoting the regression's intercept by $b$, its slope by $m$, the current step by $s$ and the relative tolerance by $ε$, the convergence condition is:
#
# $$\left\lvert\frac{(b + 2mi) - (b + mi)}{b + mi}\right\rvert = \frac{mi}{b + mi} < ε \,.$$
#
# :::{note}
# `ConstantCost` and `ConstantParams` apply different heuristics to determine what constitutes a “small” deviation.
# - `ConstantCost` assumes that the cost is akin to an unscaled log-likelihood: absolute values are irrelevant, but *absolute differences* are proportional to the log of the probability ratio.
# - `ConstantParams` assumes that values have some real, physical meaning: absolute values are meaningful, and can be used to compute *relative differences*. Absolute differences are *not* meaningful, since different parameters may scale very differently.

# %%
from scipy.stats import linregress


# %%
class ConstantParams(ConvergenceTest):
    """
    Parameters
    ----------
    param_recorder: recorder name (str)
        Parameter recorder.
        Example: 'Θ' (for `~sinnfull.optim.recorders.ΘRecorder`)
    rtol: float > 0
        Parameter values with a relative difference less than `rtol` are
        considered “equal” for the purpose of testing convergence.
    n: int ≥ 3
        Number of cost steps to use for the linear regression, starting from
        the last one.

    Returns
    -------
    `OptimizerStatus.Converged`
        - If a linear projection for twice the current number of steps predicts
          a relative difference with the current value of less than `tol`.
        - If the specified param recorder is not attached to the optimizer.
          A message stating this is also logged at level 'Error'.
    `OptimizerStatus.NotConverged`
        Otherwise
    """
    __serialized_type__: constr(regex="^ConstantParams$")="ConstantParams"
        # Used to prevent deserializing into a different subclass; each subclass must define a different name
    param_recorder: str
    rtol         : PositiveFloat=2**-6  # ~0.0156
    n            : conint(gt=2)

    def skip(self, recorders: List[Recorder], optimizer: Optimizer) -> bool:
        return any(len(rec) < self.n for rec in recorders) or super().skip(recorders, optimizer)
            # First condition: Not enough data points yet
    def __call__(self, recorders: Dict[str,Recorder], optimizer: Optimizer):
        try:
            Θrecorder = recorders[self.param_recorder]
        except KeyError:
            logger.error("Unable to apply `ConstantCost` test: no recorder "
                         f"named '{self.param_recorder}' exists.")
            return OptimizerStatus.NotConverged
        if self.skip([Θrecorder], optimizer):
            return self.last_result
        
        steps = Θrecorder.steps[-self.n:]
        if Θrecorder.keys and len(Θrecorder.keys) > 1:
            values = zip(*Θrecorder.values[-self.n:])  # Slice & transpose list of values
            keys =Θrecorder.keys
        elif Θrecorder.keys:
            values = [Θrecorder.values[-self.n:]]
            keys = Θrecorder.keys
        else:
            values = [Θrecorder.values[-self.n:]]
            keys = [Θrecorder.name]
        for valkey, value in zip(keys, values):
            value = np.array(value)
            validcs = np.ndindex(value.shape[1:])
            for validx in validcs:
                valslice = value[(slice(None),*validx)]
                res = linregress(steps, valslice)
                if res.slope == 0:
                    continue
                stepi = steps[-1]
                if abs(res.slope*stepi/(res.intercept + res.slope*stepi)) > self.rtol:
                    self.last_result = OptimizerStatus.NotConverged
                    return self.last_result
        else:
            # We've gone through all parameters, none of them triggered NotConverged
            if "Parameters converged." not in optimizer.outcome:
                logger.info("Parameters have converged: they are constant.")
                optimizer.outcome += ("Parameters converged.",)
            self.last_result = OptimizerStatus.ParamsConverged
            return self.last_result
        # We should never reach this point
        logger.error("AssertionError: ConstantParams test exited abnormally.")

ConvergenceTest._class_registry['ConstantParams'] = ConstantParams

# %% [markdown]
# :::{admonition} Further thoughts
# :class: dropdown
#
# Conceptually, we want to detect when the slope of a parameter is zero.
# The result object of `scipy.stats.linregress` provides the $p$-value for a hypothesis test where the null hypothesis is that the slope is zero; i.e.
#
# $$p = P(\text{data}|\text{slope is zero})$$
#
# It is tempting to use something $1 - p$ to detect when the probability that the slope is zero is high. This implies inverting the conditional probability, which introduces nontrivial scaling factors $p(\text{data})$, $p(\text{slope is zero})$ – concretely, this means that deciding a generally applicable threshold on the value of $p$ for declaring convergence probably not possible.
# The alternative would be to write our own hypothesis test where the null hypothesis is that the slope is different than zero, but this can also be difficult.
#
# Any condition based on a hypothesis test also suffers from an additional problem, in that statistical significance may be too sensitive. For example, in the example below, the parameter $a$ decays exponentially and after 200 steps has clearly converged. However, because there is no noise, there always remains a small, statistically significant slope – we want a criterion that will either ignore a slope when it is small enough, or detect when it is decreasing.  
# :::

# %% [markdown]
# ## Test & example

# %%
if __name__ == "__main__":
    from sinnfull.optim.recorders import LogpRecorder, ΘRecorder
    from sinnfull.optim import OptimParams
    from collections import defaultdict
    class MockOptimizer:
        stepi: int=0
        outcome: tuple=()
        Θ: OptimParams
        def logp(self):
            return 10*np.exp(-self.stepi/30)*(1+np.sin(self.stepi))

    # %%
    optimizer = MockOptimizer()
    optimizer.Θ = OptimParams(
        a=np.array([0.5, -0.5]),
        b=3.)

    Θrecorder = ΘRecorder(optimizer, keys=['a', 'b'])
    logprecorder = LogpRecorder()

    # %%
    costtest = ConstantCost(cost_recorder='log L', maximize=False, n=6)
    Θtest = ConstantParams(param_recorder='Θ', n=4)
    cost_and_Θ_test = costtest & costtest
    Θconverged = []
    costconverged = []
    cost_and_Θ_converged = []

    # %%
    def params_converged(result):
        return (result & OptimizerStatus.ParamsConverged) is OptimizerStatus.ParamsConverged
    recorders = {'log L': logprecorder, 'Θ': Θrecorder}
    
    for step in range(300):
        optimizer.stepi += 1
        optimizer.Θ.a += optimizer.stepi**-2
        optimizer.Θ.b *= (1+optimizer.stepi**-2)
        if Θrecorder.ready(optimizer.stepi):
            Θrecorder.record(optimizer.stepi, optimizer)
        if logprecorder.ready(optimizer.stepi):
            logprecorder.record(optimizer.stepi, optimizer)

        Θconverged.append(
            (optimizer.stepi, params_converged(Θtest(recorders, optimizer) )))
        costconverged.append(
            (optimizer.stepi, costtest(recorders, optimizer) is OptimizerStatus.Converged))
        cost_and_Θ_converged.append(
            (optimizer.stepi, params_converged(cost_and_Θ_test(recorders, optimizer))))                  

    # %%
    recvalues = defaultdict(lambda: [])
    pvalues = defaultdict(lambda: [])
    slopes = defaultdict(lambda: [])
    projections = defaultdict(lambda: [])
    relΔs = defaultdict(lambda: [])
    steps, all_values = Θrecorder.trace(squeeze=False)
    Δ = 4
    for valkey, values in zip(Θrecorder.keys, all_values):
        values = np.array(values)
        validcs = list(np.ndindex(values.shape[1:]))
        # Fill recorded values
        for validx in validcs:
            recvalues[(valkey, validx)] = values[(slice(None),*validx)]
        # Fill p-values, slopes
        for k in range(Δ, len(steps)):
            _steps = steps[k-Δ:k]
            _values = values[k-Δ:k]
            for validx in validcs:
                res = linregress(_steps, _values[(slice(None),*validx)])
                pvalues[(valkey, validx)].append((steps[k], res.pvalue))
                slopes[(valkey, validx)].append((steps[k], res.slope))
                project = res.intercept + res.slope*(2*steps[k])
                projections[(valkey, validx)].append((steps[k], project))
                relΔs[(valkey, validx)].append(
                    (steps[k], abs((project - _values[(-1,*validx)])/_values[(-1,*validx)])))

    # %%
    curve_cost = hv.Curve(zip(*logprecorder.trace()), kdims=['step'], vdims=['cost'])
    curves_values = hv.Overlay(
        [hv.Curve(zip(Θrecorder.steps, recvals), kdims=['step'], vdims=['value'])
         for (valkey, validx), recvals in recvalues.items()])
    curves_pvalues = hv.Overlay(
        [hv.Curve(pvalue, label=f"{valkey}, {validx}", kdims=['step'], vdims=['p value'])
         for (valkey, validx), pvalue in pvalues.items()])
    curves_slopes = hv.Overlay(
        [hv.Curve(slope, label=f"{valkey}, {validx}", kdims=['step'], vdims=['slope'])
         for (valkey, validx), slope in slopes.items()])
    curves_projections = hv.Overlay(
        [hv.Curve(project, label=f"{valkey}, {validx}", kdims=['step'], vdims=['projection'])
         for (valkey, validx), project in projections.items()])
    curves_relΔ = hv.Overlay(
        [hv.Curve(relΔ, label=f"{valkey}, {validx}", kdims=['step'], vdims=['relΔ'])
         for (valkey, validx), relΔ in relΔs.items()])

    # %%
    cciter = iter(costconverged)
    for step, status in cciter:
        if status:
            break
    assert all(status for _, status in cciter)  # Once converged, stays converged
    label_costconverged = hv.Scatter([(step, 1)], label="Cost converged").opts(size=0)
    fill_costconverged = hv.VLine(step) * label_costconverged
    fill_costconverged.opts(hv.opts.VLine(color="#dda9a9", alpha=0.85), hv.opts.Scatter(color="#dda9a9"));

    Θciter = iter(Θconverged)
    for step, status in Θciter:
        if status:
            break
    assert all(status for _, status in Θciter)  # Once converged, stays converged
    label_Θconverged = hv.Scatter([(step, 1)], label="Θ converged").opts(size=0)
    fill_Θconverged = hv.VLine(step) * label_Θconverged
    fill_Θconverged.opts(hv.opts.VLine(color="#a9a9dd", alpha=0.85), hv.opts.Scatter(color="#a9a9dd"));

    cΘciter = iter(cost_and_Θ_converged)
    for step, status in cΘciter:
        if status:
            break
    assert all(status for _, status in cΘciter)  # Once converged, stays converged
    label_cost_and_Θ_converged = hv.Scatter([(step, 1)], label="Cost & Θ converged").opts(size=0)
    fill_cost_and_Θ_converged = hv.VLine(step) * label_cost_and_Θ_converged
    fill_cost_and_Θ_converged.opts(hv.opts.VLine(color="#a9dda9", alpha=0.85), hv.opts.Scatter(color="#a9dda9"));

    # %%
    default_rtol = hv.HLine(ConstantParams.__fields__['rtol'].default
                           ).opts(color='grey', line_dash='dashed', line_width=2)
    fill_converged = fill_costconverged * fill_Θconverged * fill_cost_and_Θ_converged
    layout = (curve_cost * fill_converged
              + (curves_values * curves_projections) * fill_converged
              + curves_relΔ * default_rtol * fill_converged
              + curves_pvalues * fill_converged
              + curves_slopes * fill_converged).cols(2)
    layout.opts(hv.opts.Curve(tools=['hover'])).redim.range(step=(0,300))

# %% [markdown]
# Check that *Cost & Θ* test declares convergence only when both *cost* and *Θ* have converged.

    # %%
    (fill_costconverged + fill_Θconverged + fill_cost_and_Θ_converged).opts(
        hv.opts.VLine(width=200, height=200, clone=True),
        hv.opts.Scatter(show_legend=False, clone=True)
    ).redim.range(x=(0,300))

# %%
