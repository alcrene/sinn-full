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

# %% [markdown]
# # Testing parameter-only fits
#
# A necessary condition to fitting parameters and latent dynamics simultaneously is being able to fit parameters by themselves. A failure of this test may indicate:
#
# **Unresolved degenerate parameters**
#
# :   We know from past work that if some parameter combinations are unidentifiable, then fits are generally unable to converge. A typical symptom is fit dynamics where the parameter value increases or decrease monotonically, with no sign of slowing.
#
# **Ill-chosen hyperparameters**
#
# :   For example, any learning rate too high for fitting parameters alone, is almost certainly too high for fitting them alongside latents.

# %% [markdown]
# **FIXME**: Much of the plotting can be replaced by calls to functions in _sinnfull.viewing_. Follow pattern used in [sinnfull.view.result_viewer.ipynb](sinnfull.view.result_viewer.ipynb).

# %%
import sinnfull
sinnfull.setup('theano', view_only=True)
import sinn

# %%
from sinnfull import ureg, projectdir
from sinnfull.parameters import ParameterSet
from sinnfull.utils import generate_task_from_nb, run_as_script

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
import logging
#logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger('diagnose.fit_stability')

# %%
# Plotting
import matplotlib.pyplot as plt
from sinnfull.viewing.fit import plot_latent_evolution, plot_fit_dynamics, plot_fit_targets
from sinnfull.viewing.stream import plot_stream, add_titles, add_watermark
from sinnfull.viewing.config import pretty_names

# %%
from sinnfull.models import OUInput

model = OUInput

# %%
task_save_location = projectdir/"sinnfull/diagnose/diagnosed_task"
# Will be saved under diagnosed_task.taskdesc.json

# %%
params = [
    dict(reason="Diagnose – param-only fit",
         task_save_location = task_save_location,

         nsteps=5000,
         fit_hyperθ_updates={'params': {'λθ':1e-5}},
         #step_kwargs = {'update_latents': False},  # This is what restricts fit to parameters

         θ_init_key = θ_init_key,

         model_name = 'OUInput',
         observed_hists    =['I', 'Itilde'],
         latent_hists      =[],  # Observed latents <=> using ground truth latents
         #parameter_gen_kwargs = {'M': 2, 'Mtilde': 2},
         default_objective = objectives[model_name].logp_forward_se,
         nodyn_objective = objectives[model_name].logp_nodyn_se,
         default_model_params = model_params[model_name].unequal_σ,
         default_learning_params = learning_params[model_name].default,
         prior = priors[model_name](M=2, Mtilde=2)
        )
    # The key prefixes 1-4 are used for other RNGs
    for θ_init_key in [(5,i) for i in range(1)]
]

# %%

# %%
tasks = []
for p in params[]:
    generate_task_from_nb(
        projectdir/"sinnfull/run/Optimize_WF_template.ipynb",
        parameters=p
    );
    tasks.append(smttask.Task.from_desc(
        task_save_location.with_suffix(".taskdesc.json")))

# %%
# Get ground truth data from the first task
synth_data = tasks[0].optimizer.data_segments.data.run()
ground_truth = model.remove_degeneracies(
    synth_data.trials.trial.data[0].params.get_values())

# %%
fig = plt.figure(figsize=(6, 4.5), constrained_layout=True)
axes = fig.axes
for task in tasks:
    for recorder in task.run()[2]:
        if recorder.name == 'Θ':
            break
    assert recorder.name == 'Θ'
    recorder.drop_step(0)  # The first step is recorded before parameters are normalized
    fig, axes = plot_fit_dynamics(recorder, exclude={'M', 'Mtilde'},
                                  fig=fig, axes=axes,
                                  color='#BBBBBB')
for ax in axes['τtilde']:
    ax.set_yscale('log')
plot_fit_targets(axes, ground_truth);

# %%
from sinnfull.viewing.record_store_viewer import RSView

# %%
#records = reclist.filter.on(20201028).list
#records = reclist.filter.on(20201102).list
#records = reclist.filter.on(20201103).filter.reason('OU2').list
#recordsOU2 = reclist.filter.after((2020,11,3,15)).filter.reason('OU2').list
#recordsOU3 = reclist.filter.after((2020,11,3,15)).filter.reason('OU3').list
recordsOU2 = reclist.filter.on((2020,11,4)).filter.reason('OU2').list
recordsOU3 = reclist.filter.on((2020,11,4)).filter.reason('OU3').filter(lambda rec: len(rec.outputpath) == 5).list  # Hash collision caused two runs to be same record

# %%
records = reclist.filter.after(20201109).filter.reason("Test fit").filter.reason("Observed").list

# %%
rsview = RSView().filter.tags('__finished__') \
          .filter.after(20210324).filter.reason("fit stability").list

# %%
task.optimizer.data_segments.data

# %%
# Get ground truth data from the first task
task = smttask.Task.from_desc(records[0].parameters)
synth_data = task.optimizer.data_segments.data
if hasattr(synth_data, 'run'):
    synth_data = synth_data.run()
ground_truth = model.remove_degeneracies(
    synth_data.trials.trial.data[0].params)

# %%
#fig = plt.figure(figsize=(7, 4.5), constrained_layout=True)
fig = plt.figure(figsize=(12, 7.75), constrained_layout=True)
axes = fig.axes
λθ = 1e-4
for rec in records:
    task = smttask.Task.from_desc(rec.parameters)
    if λθ != task.optimizer.fit_hyperparams['params']['λθ']:
        continue
    recorder = rec.get_output('Θ')
    recorder.drop_step(0)  # The first step is recorded before parameters are normalized
    fig, axes = plot_fit_dynamics(recorder, exclude={'M', 'Mtilde'},
                                  fig=fig, axes=axes,
                                  color='#BBBBBB')
for ax in axes['τtilde']:
    ax.set_yscale('log')
plot_fit_targets(axes, ground_truth);

for ax in axes['Wtilde']:
    ax.set_ylim(-7.5,7.5)

fig.suptitle(f"Fit stability test – OUInput – logp_forward – $λ^θ$: {λθ}");

# %%
fig = plt.figure(figsize=(7, 4.5), constrained_layout=True)
axes = fig.axes
for rec in records:
    recorder = rec.get_output('Θ')
    recorder.drop_step(0)  # The first step is recorded before parameters are normalized
    fig, axes = plot_fit_dynamics(recorder, exclude={'M', 'Mtilde'},
                                  fig=fig, axes=axes,
                                  color='#BBBBBB')
plot_fit_targets(axes, ground_truth);

# %%
from mackelab_toolbox.parameters import ParameterSet, ParameterComparison

# %%
pdiff = ParameterComparison((ParameterSet(records[0].parameters).inputs,
                             ParameterSet(tasks[0].json())),
                            labels=("record", "params"))

# %%
df.loc[:,[c for c in df.columns if 'digest' not in c]]

# %%
with open("diagnosed_task2.taskdesc.json", 'w') as f:
    f.write(records[0].parameters)

# %%
record = records[0]
recorders = {'latents': record.get_output('latents'),
             'Θ': record.get_output('Θ')}

# %%
#TODO: Should be done by Pydantic
from mackelab_toolbox.typing import Array
recorder = recorders['latents']
recorder.values = [[Array['float64'].validate(v) for v in vallist]
                   for vallist in recorder.values]

# %%
rec_task = smttask.Task.from_desc(record.parameters)
optimizer = rec_task.run()[1]

# %%
target_data = next(optimizer.data_segments)[2]

# %%
plot_latent_evolution(recorders,
                      col_steps=[1, 5, 5000],
                      hist_names=['Itilde', 'I'],
                      rows_per_hist=2,
                      model=optimizer.model,
                      target_data=target_data,
                      ploth=1.5, plotw=3
                     );

# %%
param_task = tasks[0]
optimizer = param_task.run()[1]
recorders = {r.name: r for r in param_task.run()[2]}
target_data = next(optimizer.data_segments)[2]

# %%
plot_latent_evolution(recorders,
                      col_steps=[1, 5, 5000],
                      hist_names=['Itilde', 'I'],
                      rows_per_hist=2,
                      model=optimizer.model,
                      target_data=target_data,
                      ploth=1.5, plotw=3
                     );

# %%
param_task.optimizer.data_segments.draw(figsize=(8,4.5))

# %%
rec_task.optimizer.data_segments.draw(figsize=(8,4.5))

# %%
target_data.Itilde.data[:10]

# %%
target_data.Itilde.data[:10] * optimizer.model.σtilde.get_value()

# %%
optimizer.model.Itilde.data[:10]

# %%
optimizer.model.σtilde.get_value()

# %%
optimizer.data_segments.data.trials.trial.data[0].params.σtilde

# %%
optimizer.model.Itilde.data[:10]

# %%
optimizer.data_segments.data.trials

# %%
optimizer.data_segments.data.trials.trialkey.data[0]

# %%
optimizer.data_segments.data.trials.trial.data[0]

# %%
optimizer.data_segments.data.trials.trial.data[0].param_hash

# %%
optimizer.data_segments.data.trials.trial.data[0].params.get_values()

# %%
trial = optimizer.data_segments.data.trials.trial.data[0]
optimizer.data_segments.data.load(trial)

# %%
optimizer.data_segments.data.load(optimizer.data_segments.data.trials.trial.data[0])

# %%
