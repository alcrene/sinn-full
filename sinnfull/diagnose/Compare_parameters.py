# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Comparing run parameters
#
# A common situation is that we have one or multiple task parameterizations, and we want to know how they differ. For example, after a parameter sweep, to see to which parameters the task is most/least sensitive. Or, if we think a task should already have been run but *smttask* seems to want to run it again, we can compare the parameterizations of the target task and the one we think was run already.
#
# The challenge is that tasks can depend on upwards of 100 parameters, organized in an arbitrary hierarchy. Simply printing both sets and comparing by eye isn't going to suffice. Instead, we use *mackelab_toolbox*'s `ParameterComparison` object to do the comparisons and present differing values in a Pandas DataFrame.

# %%
import sinnfull
sinnfull.setup('theano')
import sinn

# %%
from sinnfull import ureg, projectdir
#from sinnfull.parameters import ParameterSet
from sinnfull.utils import generate_task_from_nb
from smttask.utils import get_task_param

import numpy as np
from devtools import debug
from sinnfull.parameters import ParameterSet
import mackelab_toolbox as mtb
from mackelab_toolbox.parameters import ParameterComparison

import os
os.chdir(projectdir/"sinnfull/diagnose")

# %%
import smttask

# %%
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    smttask.config.record = False

# %%
records = smttask.view.RecordList()

# %% [markdown]
# ## Comparing parameters from multiple recorded runs

# %%
# Compare two specific task descriptions
diff = ParameterComparison(
    [
     #"../workflows/tasklist/n5000/OptimizeModel__e0be29e0c3__nsteps_5000.taskdesc.json",
     "../view/tasklist/OptimizeModel__1d3ce59df6__nsteps_5000.taskdesc.json",
     "../view/tasklist/OptimizeModel__2.taskdesc.json"
    ],
    labels=["orig", "re-generated"])

# %%
# Compare all runs which occurred on 24 Oct 2020
diff = ParameterComparison(rec for rec in records.filter.on(20201024))

# %% [markdown]
# We exclude “reason” here from the displayed params because it is free-form text, and not used to compute digests.
# The symbol `<+>` indicates

# %%
diff.dataframe(depth=5)#.drop(columns='reason')

# %%
get_task_param(diff.records[1].parameters, 'optimizer.prior_params')

# %%
get_task_param(diff.records[0].parameters, 'optimizer.prior_params')

# %% [markdown]
# ## Comparing a recorded run with a new task description
#
# Motivating example: A task was run on a remote machine. Now on the local machine, we define the task again, and expect that when executing it, the result from the previous run is found and returned. Instead, the task is being re-executed.
#
# This indicates that the digests for the previous and current tasks don't match, and we would like to know why (perhaps we forgot about a parameter we changed ?)

# %%
task_save_location = projectdir/"sinnfull/diagnose/diagnosed_task"
# Will be saved under diagnosed_task.taskdesc.json

# %% [markdown]
# Define a task in the same way we would prepare a *taskdesc* file for execution.

# %%
params = [
    dict(reason="Diagnose – param-only fit",
         task_save_location = task_save_location,

         nsteps=5000,
         fit_hyperθ_updates={'params': {'λθ':1e-5}},
         step_kwargs = {'update_latents': False},  # This is what restricts fit to parameters

         θ_init_key = θ_init_key,
         latents_init_key = 'ground truth',        # Latents must be initialized with ground truth for this test

         model_name = 'OUInput',
         observed_hists    =['I'],
         latent_hists      =['Itilde'],
         parameter_gen_kwargs = {'M': 2, 'Mtilde': 2}
        )
    # The key prefixes 1-4 are used for other RNGs
    for θ_init_key in [(5,i) for i in range(1)]
]

# %% [markdown]
# To better ensure consistency, we run the task generation notebook, and read the *taskdesc* file back from disc.

# %%
generate_task_from_nb(
    projectdir/"sinnfull/run/Optimize_WF_template.ipynb",
    parameters=params[0]
);

# %%
task = smttask.Task.from_desc(task_save_location.with_suffix(".taskdesc.json"))

# %%
params_task = ParameterSet(task.json())

# %% [markdown]
# From the runs executed on a given date, we find one which matches some of the parameters we sweeped

# %%
for rec in records.filter.on(20201024):
    params_rec = ParameterSet(rec.parameters).inputs
    if np.isclose(params_rec.optimizer.inputs.model.params.μtilde[0],
                  params_task.optimizer.inputs.model.params.μtilde[0]):
        print(rec.label)
        break

# %% [markdown]
# Create the comparison object

# %%
diff = mtb.parameters.ParameterComparison((params_rec, params_task), ('record', 'task'))

# %%
diff.dataframe(depth=5)

# %% [markdown]
# So the model data differs, and that's why we have a different digest. But is it really different ? Print the summary for the `I` data from each record, they look identical:

# %%
print("Recorded task: I.data summary:")
print(params_rec.optimizer.inputs.model.I.data[1]['summary'])
print()
print("New task: I.data summary:")
print(params_task.optimizer.inputs.model.I.data[1]['summary'])

# %% [markdown]
# But the actual compressed data have different lengths:

# %%
print("Recorded task: len(I.data) =",
      len(params_task.optimizer.inputs.model.I.data[1]['data']))
print("New task:      len(I.data) =",
      len(params_rec.optimizer.inputs.model.I.data[1]['data']))

# %% [markdown]
# Even though the values unpack to exactly the same array:

# %%
from mackelab_toolbox.typing import Array
A_rec = Array[np.float64].validate(params_rec.optimizer.inputs.model.I.data)
A_task = Array[np.float64].validate(params_task.optimizer.inputs.model.I.data)
print("Both data arrays equal:", np.all(A_rec == A_task))

# %% [markdown]
# In this case the difference is due to the compression being different on remote and local machines. *smttask* now deactivates compression when computing digests, to avoid this issue.
