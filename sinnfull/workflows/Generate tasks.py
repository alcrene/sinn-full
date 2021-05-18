# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metada_filter: -jupytext.text_representation.jupytext_version
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
# # Task generator
#
# Execute the [*Optimize_WF_template* notebook](Optimize_WF_template.ipynb) with a list of parameters to generate a task list.
#
# Each execution generates a _task file_, which can then be executed from the command line.
#
# :::{admonition} Reminder  
# We use the [convention](../optim/base) that
#   - $θ$ refers to _parameters_,
#   - while $η$ refers to _latent variables_.
#
# Among other things, these are used as prefixes to differentiate hyperparameters.  
# :::

# %%
import sinnfull
sinnfull.setup()

# %%
from itertools import product
from tqdm.auto import tqdm
from sinnfull import ureg, projectdir
from sinnfull.parameters import ParameterSet
from sinnfull.utils import generate_task_from_nb, model_name_from_selector
from pydantic import BaseModel

import os
os.chdir(projectdir/"sinnfull/workflows")

# %%
from sinnfull.models import paramsets, objectives, priors
from sinnfull.optim import paramsets as optim_paramsets

# %%
# The directory to which we will save task files
task_save_location = "tasklist"

if not os.path.exists("tasklist"):
    os.mkdir("tasklist")

# %% [markdown]
# ## Fully observed Wilson-Cowan model
# The example below fits a Wilson-Cowan $α^{-1} \frac{d}{dt}{u}_t = L[{u}_t] + {w} F_{β,h}[{u}_t] + I_t$, where the sequence $I_t$ (a white noise process) is _known_. The unknown model parameters are $α$, $β$, $h$ and $w$.
#
# For more details, see [the model's notebook](../models/WC/WC).

# %%
n_fits         = 11  # Number of different initial conditions to draw
θ_learning_rates = [0.0002]
θ_clip         = 100.

# %%
model_selector = {"__root__": {"ObservedDynamics"},
                  "input"   : {"GaussianWhiteNoise"},
                  "dynamics": {"WilsonCowan"},
                  "__connect__": {'GaussianWhiteNoise.ξ -> WilsonCowan.I'}}

objective_selectors = {'input': {'GaussianWhiteNoise'},
                       'dynamics': {'WilsonCowan', 'se'}}

default_learning_params = optim_paramsets['WC'].default 

prior_spec = ParameterSet(
    {'input'   : {'selector': ('GWN', 'default'),
                  'kwds': dict(mu_mean=[-0.25, -0.5],
    #{'input'   : {'selector': ('GWN', 'fixed_mean'),
    #              'kwds': dict(mu=[-0.25, -0.5],
                               logsigma_mean=[-1., -1.],
                               M=2)},
     'dynamics': {'selector': ('WC', 'rich'),
                  'kwds': dict(M=2)}
    })
synth_param_spec = prior_spec.copy()  # Requires sinnfull.ParameterSet
synth_param_spec.update({'input.kwds.mu_std': 1.,         # Tip: Use dotted notation to avoid
                         'input.kwds.logsigma_std': 0.5,  # quashing other params
                         'dynamics.kwds.scale': 0.25})

# %%
params = [
    dict(reason=f"Example - {model_name_from_selector(model_selector)}",
         task_save_location = task_save_location + f"/n{nsteps}",
         # Fit
         nsteps=nsteps,
         default_learning_params = default_learning_params,
         fit_hyperθ_updates={'params': {'λθ':λθ, 'clip': θ_clip, 'b1': 1., 'b2': 1.}},
         Θ_init_key = Θ_init_key,
         model_rngkey = latent_init_key,  # Determines initialization of the latent
         optimizer_rngkey= 2,    # Affects drawn batch samples during iterations
         sampler_rngkey  = 5,    # Affects the data segment drawn for each iteration
         # Data
         synth_param_spec = synth_param_spec,
         param_rngkey    = 3,    # Affects the parameters drawn for the synthetic dataset
         sim_rngkey      = 4,    # Affects the model integrator used to generate synthetic data
         # Model
         model_selector      = model_selector,
         observed_hists      =["dynamics.u", "input.ξ"],
         latent_hists        =[],
         objective_selectors = objective_selectors,
         prior_spec          = prior_spec,
        )
    for Θ_init_key, latent_init_key in 
        [((6,i), (1,i)) for i in range(n_fits)] + [('ground truth', (1,0))]
    for λθ in θ_learning_rates
    for nsteps in [5000]
]
for p in params:
    if p['latent_hists']:
        p['task_save_location'] += "-latents"

# %% [markdown]
# ## Wilson-Cowan model with white noise latent (unknown) input
#
# Same model as [above](#fully-observed-wilson-cowan8model), with the difference that the sequence $I_{0:T}$ (a white noise process) is now _unknown_. It must thus be inferred, in the form of a long vector on length $T$, along with the model parameters $α$, $β$, $h$ and $w$.
#
# The differences with the fully observed case:
#
# - We add fit hyperparameters for the latent variables (prefixed with `η`).
# - `"dynamics.I"` is now listed as a _latent_ history.[^1]
#
# [^1] We could use `"input.ξ"` instead of `"dynamics.I"`; the two point to the same history.

# %%
n_fits         = 11  # Number of different initial conditions to draw
θ_learning_rates = [0.0002]
η_learning_rates = [0.001]
θ_clip         = 100.
η_clip         = 100.

# %%
params = [
    dict(#reason=f"Example - {model_name_from_selector(model_selector)}",
         reason=f"Test: random latent init - {model_name_from_selector(model_selector)}",
         task_save_location = task_save_location + f"/n{nsteps}",
         # Fit
         nsteps=nsteps,
         default_learning_params = default_learning_params,
         fit_hyperθ_updates={'params': {'λθ':λθ, 'clip': θ_clip, 'b1': 1., 'b2': 1.}},
         Θ_init_key = Θ_init_key,
         model_rngkey = latent_init_key,  # Determines initialization of the latent
         optimizer_rngkey= 2,    # Affects drawn batch samples during iterations
         sampler_rngkey  = 5,    # Affects the data segment drawn for each iteration
         # Data
         synth_param_spec = synth_param_spec,
         param_rngkey    = 3,    # Affects the parameters drawn for the synthetic dataset
         sim_rngkey      = 4,    # Affects the model integrator used to generate synthetic data
         # Model
         model_selector      = model_selector,
         observed_hists      =["dynamics.u"],
         latent_hists        =["dynamics.I"],
         objective_selectors = objective_selectors,
         prior_spec          = prior_spec,
        )
    for Θ_init_key, latent_init_key in 
        [((6,i), (1,i)) for i in range(1,n_fits+1)] + [('ground truth', (1,0))]
    for λθ in θ_learning_rates for λη in η_learning_rates
    for nsteps in [25000]
]
for p in params:
    if p['latent_hists']:
        p['task_save_location'] += "-latents"

# %% [markdown]
# ## Create the tasks
#
# Run this after creating of the task lists above.

# %% [markdown]
# ::::{tip}  
# The most efficient way to debug task creation is to run [*Optimize_WF_template*](Optimize_WF_template.ipynb) with exactly the parameters that caused the error. You can obtain these parameters with the following:
#
# ```python
# import sinnfull.utils
# sinnfull.utils.papermill_parameter_block(params[0])
# ```
#
# Then just paste the returned code block _below_ the parameter block into the *Optimize_WF_template* notebook and run it.  
# (Change the value *exec_environment* to `'notebook'` to actually run the created Task.)  
# ::::

# %% [markdown]
# import sinnfull.utils
#
# sinnfull.utils.run_as_script("sinnfull.workflows.Optimize_task_template",
#                              **params[0])

# %%
for p in tqdm(params):
    generate_task_from_nb(
            "Optimize_WF_template.ipynb",
            parameters=p,
            return_val='none'
        )

# %% [markdown]
# Creating tasks in parallel is unfortunately still a WIP; eventually it should like as follows:
#
#     from multiprocess import Pool
#     import functools
#
#     with Pool(4) as pool:
#         def worker(p):
#             return generate_task_from_nb(
#                 "Optimize_WF_template.ipynb",
#                 parameters=p,
#                 return_val='none'
#             )
#         worklist = pool.imap(worker, params)
#         for job in tqdm(worklist, total=len(params)):
#             pass

# %%
