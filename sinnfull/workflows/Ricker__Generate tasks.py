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
# These are for example used as prefixes to differentiate hyperparameters.  
# :::

# %%
import sinnfull
sinnfull.setup()

# %%
from itertools import product
from tqdm.auto import tqdm
from sinnfull import ureg, projectdir
from sinnfull.parameters import ParameterSet
from sinnfull.utils import generate_task_from_nb
from pydantic import BaseModel

import os
os.chdir(projectdir/"sinnfull/workflows")

# %%
from sinnfull.models import model_params, objectives, priors
from sinnfull.optim import learning_params

# %%
# Create the directory to which we will save tasks
if not os.exists("tasklist"):
    os.mkdir("tasklist")

# %% [markdown]
# ## Fully observed model
# The example below fits a Ricker map $N_{t+1} = r N_t e^{-N_t/\varphi + e_t}$, where the sequence $e_t$ is _known_. The only unknown parameters are $r$ and $\varphi$.
#
# For more details, see [the model's notebook](../models/ricker.ipynb).

# %%
model_name     = "Ricker"
n_fits         = 35  # Number of different initial conditions to draw
θ_learning_rates = [0.0002]
θ_clip         = 100.
task_save_location = "tasklist"
# # mkdir tasklist
params = []

# %%
prior = priors[model_name].default(M=1)
params = [
    dict(reason=f"Example - {model_name} model"
         task_save_location = task_save_location + f"/n{nsteps}",
         nsteps=nsteps,
         fit_hyperθ_updates={'params': {'λθ':λθ, 'clip': θ_clip, 'b1': 1., 'b2': 1.},
                             'latents':{'λη':λη, 'clip': η_clip}},
         θ_init_key = θ_init_key,
         model = model_name,
         observed_hists    =["N", "e"],
         latent_hists      =[],
         default_objective = objectives[model_name].logp_forward,
         default_model_params = model_params[model_name].default,
         default_learning_params = learning_params[model_name].default,
         prior = prior
        )
    for θ_init_key in [(5,i) for i in range(n_fits)] + ['ground truth']
    for λθ in θ_learning_rates for λη in η_learning_rates
    for nsteps in [5000]
]

# %% [markdown]
# ## Model with white noise latent
#
# This is the form of the Ricker map studied by Wood (2010), where $e$ is unobserved white noise. The sequence of values $e_{0:T}$ is inferred along with the parameters $r$, $\varphi$ and $σ$.
#
# The differences with the fully observed case:
#
# - We add fit hyperparameters for the latent variables (prefixed with `η`).
# - `"e"` is now listed as a _latent_ history.

# %%
model_name     = "Ricker"
n_fits         = 35  # Number of different initial conditions to draw
θ_learning_rates = [0.0002]
η_learning_rates = [0.001]
θ_clip         = 100.
η_clip         = 100.
task_save_location = "tasklist"
# # mkdir tasklist
params = []

# %%
prior = priors[model_name].default(M=1)
params = [
    dict(reason=f"Example - {model_name} model"
         task_save_location = task_save_location + f"/n{nsteps}",
         nsteps=nsteps,
         fit_hyperθ_updates={'params': {'λθ':λθ, 'clip': θ_clip, 'b1': 1., 'b2': 1.},
                             'latents':{'λη':λη, 'clip': η_clip}},
         θ_init_key = θ_init_key,
         model = model_name,
         observed_hists    =["N"],
         latent_hists      =["e"],
         default_objective = objectives[model_name].logp_forward,
         default_model_params = model_params[model_name].default,
         default_learning_params = learning_params[model_name].default,
         prior = prior
        )
    for θ_init_key in [(5,i) for i in range(n_fits)] + ['ground truth']
    for λθ in θ_learning_rates for λη in η_learning_rates
    for nsteps in [5000]
]

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
# ::::

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
