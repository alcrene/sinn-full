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

# %% [markdown]
# # Searching tagged collections
#
# These utility functions can be used construct objects from specifiers for
# the tagged collections `models`, `priors` and `objectives`.
# The specifiers build are composed of selectors for `TaggedCollection`, with some additional collection-specific logic.

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import sinnfull
    sinnfull.setup()

# %% tags=["hide-input"]
from typing import Type, Union, Optional, Set, List, Tuple, Dict
from collections.abc import Collection, Callable
from functools import partial
from pydantic import StrictStr
from sinnfull.parameters import ParameterSet
from .base import ObjectiveFunction, Prior, Model
from ._scandir import models, objectives, priors, paramsets

# %%
__all__ = ["ModelSpec",
           "get_prior", "get_objectives", "get_model_class"]


# %% [markdown]
#
# These types can be used in type hints for Tasks and functions expecting
# a specifier for one of the functions above

# %%
# The Model constructor is actually defined as a Task: `CreateModel`
ModelSpec = Union[Dict[StrictStr,Union[Dict[StrictStr,Tuple[StrictStr,...]],
                       Tuple[StrictStr,...]]],
                  Tuple[StrictStr,...]]

# %% [markdown]
#
# Specifier utilities
# - `get_objectives`
# - `get_prior`
# - `get_model_class`

# %%
def get_objectives(objective_selectors: list) -> List[ObjectiveFunction]:
    """
    This function looks complicated, but it's just a series of `if` statements
    for the different formats objective selectors can take.
    Accepts a list of objective selectors, and returns the objective
    corresponding to each. The returned objectives can be combined with `sum`.
    
    Objectives selectors are either tuples or sets of tags (not lists),
    optionally including a dictionary of keyword arguments.
    Objectives for submodels are indicated by placing them in nested dictionaries.
    At any nesting level, multiple objectives can be defined by wrapping them
    in a list.
    
    Objectives are returned as a flat list, independent of nesting level.
    """
    objective_list = []
    if not isinstance(objective_selectors, list):
        objective_selectors = [objective_selectors]
    for sel in objective_selectors:
        # FIXME: It's still a bit too inconvenient to always make an ObjectiveFunction,
        #     so we sometimes still just pass a callable. But this will miss errors.
        if isinstance(sel, Callable):
            objective_list.append(sel)
        elif isinstance(sel, dict):
            # Selector for submodel
            for submodel, subsel in sel.items():
                if isinstance(subsel, (list, dict)):
                    # List of selectors for submodel => recurse and prepend `submodel`
                    for objective in get_objectives(subsel):
                        if objective.submodel:
                            submodel += "."
                        objective.submodel = submodel + objective.submodel
                            # Assumes that ObjectiveFunction.submodel defaults to ""
                        objective_list.append(objective)
                else:
                    if isinstance(subsel, Collection):
                        # dictionaries -> kwargs; everything else -> selector
                        kwargs = {k:v for e in subsel if isinstance(e, dict) for k,v in e.items()}
                        subsel = type(subsel)(e for e in subsel if not isinstance(e, dict))
                    objective = objectives[subsel].copy(deep=True)
                    if kwargs:
                        objective = type(objective)(
                            func=partial(objective.func.func, **kwargs), tags=objective.tags)
                    if objective.submodel:
                        submodel += "."
                    objective.submodel = submodel + objective.submodel
                    objective_list.append(objective)
        else:
            if isinstance(sel, Collection):
                # dictionaries -> kwargs; everything else -> selector
                kwargs = {k:v for e in sel if isinstance(e, dict) for k,v in e.items()}
                sel = type(sel)(e for e in sel if not isinstance(e, dict))
            objective = objectives[sel].copy(deep=True)
            if kwargs:
                # TODO: Objective function should provide a `partial` method
                objective = type(objective)(
                    func=partial(objective.func.func, **kwargs), tags=objective.tags)
            objective_list.append(objective)
                # Copy required to avoid modifying the instance in objectives[…]
                # (deep=True required to also copy objective.func)
    return objective_list

def get_prior(model_class: Type[Model], prior_spec: dict) -> Prior:
    """
    model_class: The model class for which to build the prior.
    prior_selector:
        dict: priors for submodels.
              Must have an entry 'selector'.
              May have an entry 'kwds'.
    """
    prior_spec = ParameterSet(prior_spec)
    with Prior() as prior:
        # TODO: Expose a public attribute on the model class
        #      (.nested_models is only defined on instances)
        # NB: Serialization depends on the order in which priors are defined
        for submodel_nm in sorted(model_class._model_identifiers):
            subprior = prior_spec[submodel_nm]
            priors[subprior.selector](**subprior.kwds, name=submodel_nm)
    return prior

def get_model_class(model_spec: ModelSpec) -> Type[Model]:
    """
    Return the model *class* corresponding to `mode_spec`.
    (Model *instances* are constructed with the task `CreateModel`.)
    """

    if isinstance(model_spec, (set,str)):
        return models[model_spec]
    else:
        return models[model_spec['__root__']]
