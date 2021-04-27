# -*- coding: utf-8 -*-
# %% [markdown]
# # Diagnostic utilities
#
# The [optimization](../optim/base) module defines a number of hooks which are not needed for
# execution, but are useful for diagnosing issues [^1]. Some diagnostic functions
# require these hooks. Since they may consume non-negligible CPU or memory, the
# are disabled by default. To enable them, use:
#
# ```python
# import sinnfull.diagnostics
# sinnfull.diagnostics.enable()
# ```
#
# Make sure to execute `enable()` before the `Optimizer` is created.
#
# [^1]: Currently these hooks are implemented directly within the [`AlternatedSGD`](../optim/optimizers/alternated_sgd) class.

# %%
import sinnfull.config as config

# %%
# Functions to turn diagnostic hooks on and off
def enable():
    config.diagnostic_hooks = True
def disable():
    config.diagnostic_hooks = False
def set(value: bool, smttask_record=config._NoValue):
    """
    Turn on (off) diagnostics. Currently this does two things:

    - Set `sinnfull.config.diagnostic_hooks` to True.
      + Tells `SGDOptimizer` to store extra values (e.g. the gradient values)
    - Turns off recording for smttask.
      + In case this is not desired, the `smttask_record` argument can be set to
        True (set `smttask.config.record` to True)
        or None (leave `smttask.config.record` unchanged)
    """
    if smttask_record is config._NoValue:
        if value:
            # If we turn on diagnostics, we definitely don't want to record
            smttask_record = False
        else:
            # If value is 'off', we can't know if record vs no-record has
            # already been set, so do nothing
            smttask_record = None
    if value is not None:  # Use None as a value which doesn't change the flag
        config.diagnostic_hooks = bool(value)
    if smttask_record is not None:
        import smttask
        smttask.config.record = smttask_record
