# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: Python (sinnfull)
#     language: python
#     name: sinnfull
# ---

# # Workflow parameters

from warnings import warn
from math import prod
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple

from mackelab_toolbox.parameters import ComputedParams

from sinnfull.parameters import ParameterSet, ParameterRange, ParameterSpace
from sinnfull.models import model_params, objectives
from sinnfull.optim import learning_params

__all__ = ['WorkflowParams', 'OptimizeWorkflowParams']

@dataclass
class WorkflowParams(ComputedParams):
    """
    A `WorkflowParams` object is used to define the arguments expected by a
    `task generation notebook`. There should be one `WorkflowParams` subclass
    for each task generation notebook.

    Since `WorkflowParams` inherits from `mackelab_toolbox.parameters.ComputedParams`,
    all subclass automatically provide functionality for

    - defining ensembles of parameters;
    - iterating over these ensembles;
    - printing a summary of expected parameters in an interactive session.
    """
    # Ensure we use our customized ParameterSet
    ParameterSet = ParameterSet
    ParameterSpace = ParameterSpace

    # Task specification
    reason          : str

@dataclass
class OptimizeWorkflowParams(WorkflowParams):
    # Some arguments can be specified as strings; compute_params() then
    # recovers the object associated with that name and model.

    # Model specification
    model_name      : str
    observed_hists  : list
    latent_hists    : list
    parameter_gen_kwargs: dict
    default_model_params: ParameterSet  # Defaults for params not returned by generator

    # Optimization specification
    nsteps          : int
    n_fits          : int
    fit_hyperθ_updates: dict
    ## Optimization objectives
    nodyn_objective  : Callable
    default_objective: Optional[Callable]=None
    params_objective : Optional[Callable]=None
    latents_objective: Optional[Callable]=None
    latents_nodyn_objective: Optional[Callable]=None

    # Defaults
    step_kwargs     : dict=field(default_factory=lambda: {})
    task_save_location: str="tasklist"
    default_learning_params: ParameterSet='default' # fit_hyperθ_updates takes precedence

    # Dynamically set parameters
    θ_init_key      : Tuple[int]=()
    # - reason (compute_params() appends model_name and latents)

    # These field names are removed before creating a ParameterSet
    non_param_fields = ('n_fits',)

    def __post_init__(self):
        "Perform some parameter validation"
        if (self.default_objective is None and
            (self.params_objective is None or self.latents_objective is None)):
            raise TypeError("Missing either a default objective, or objectives "
                            "for both prams and latents.")

    def compute_params(self):
        self.θ_init_key = ParameterRange([(5,i) for i in range(self.n_fits)])
        self.reason += (f"\nModel: {self.model_name}\n"
                        f"\nLatents: {self.latent_hists}\n")
        # Convert options which were passed as strings
        if isinstance(self.default_model_params, str):
            self.default_model_params = model_params[self.model_name][self.default_model_params]
        if isinstance(self.default_learning_params, str):
            self.default_learning_params = learning_params[self.model_name][self.default_learning_params]
        for objective_name in ['default_objective', 'params_objective',
                               'latents_objective',
                               'nodyn_objective', 'latents_nodyn_objective']:
            objective = getattr(self, objective_name, None)
            if isinstance(objective, str):
                setattr(self, objective_name,
                        objectives[self.model_name][objective])
