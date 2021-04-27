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
#     display_name: Python (sinnfull)
#     language: python
#     name: sinnfull
# ---

# %% [markdown]
# # Optimize workflow template
#
# Executing this notebook generates **1** _Task_, which can then either be executed or saved as a _task file_. A _task file_ is a JSON file containing all the information required to run a sequence of _Tasks_: the name of each _Task_ and its input parameters.
#
# :::{tip}  
# To generate a list of tasks, use the [*Generate tasks* notebook](./Generate%20tasks.py). It will execute this notebook, allowing all parameters in the *parameters cell* to be changed, and save the result as a _task file_.  
# :::

# %% [markdown]
# :::{figure-md} optimize-wf-flowchart  
# <img src="optimize-wf-flowchart.svg" title="Flowchart – Optimize Task workflow">
#
# Flowchart – Optimize Task workflow
# <a href="https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZmxvd2NoYXJ0IFREXG4gICAgc3ViZ3JhcGggd29ya2Zsb3dbT3B0aW1pemUgVGFzayBXb3JrZmxvd11cbiAgICBkaXNrZGF0YVtvbi1kaXNrIGRhdGFdXG4gICAgc3ludGhkYXRhW3N5bnRoZXRpYyBkYXRhXVxuICAgIGNkYXRhe3tjcmVhdGUgZGF0YSBhY2Nlc3Nvcn19XG4gICAgY21vZGVse3tjcmVhdGUgbW9kZWx9fVxuICAgIGNwcmlvcihbY3JlYXRlIHByaW9yXSlcbiAgICBjaW5pdChbY2hvb3NlIGluaXRpYWwgcGFyYW1zXSlcbiAgICBjb2JqKFtjaG9vc2Ugb2JqZWN0aXZlXSlcbiAgICBjaHlwZXIoW2Nob29zZSBoeXBlcnBhcmFtdGVyc10pXG4gICAgY29wdGltaXplcnt7Y3JlYXRlIG9wdGltaXplcn19XG4gICAgY3JlY3t7Y3JlYXRlIHJlY29yZGVyc319XG4gICAgY3Rlc3QoW2NyZWF0ZSBjb252ZXJnZW5jZSB0ZXN0c10pXG4gICAgY29wdGltaXple3tvcHRpbWl6ZSBtb2RlbH19XG4gICAgZGlza2RhdGEgLS4tPiBjZGF0YVxuICAgIHN5bnRoZGF0YSAtLi0-IGNkYXRhXG4gICAgY3ByaW9yIC0uLT4gc3ludGhkYXRhXG4gICAgY21vZGVsIC0uLT4gc3ludGhkYXRhXG4gICAgY3ByaW9yIC0tPiBjb3B0aW1pemVyXG4gICAgY2h5cGVyIC0uLT52aHlwZXIoW3ZhbGlkYXRlIGh5cGVycGFyYW1ldGVyc10pXG4gICAgY2RhdGEgJiBjaW5pdCAmIGNtb2RlbCAmIGNvYmogJiB2aHlwZXIgLS0-IGNvcHRpbWl6ZXJcbiAgICBjb3B0aW1pemVyICYgY3JlYyAmIGN0ZXN0IC0tPiBjb3B0aW1pemVcbiAgICBlbmRcblxuICAgIHN0eWxlIGNvcHRpbWl6ZSBmaWxsOiNjZGU0ZmYsIHN0cm9rZTojMTQ3ZWZmLCBzdHJva2Utd2lkdGg6MnB4XG4gICAgc3R5bGUgd29ya2Zsb3cgZm9udC13ZWlnaHQ6Ym9sZFxuIiwibWVybWFpZCI6e30sInVwZGF0ZUVkaXRvciI6ZmFsc2V9">
#     (edit)
# </a>  
# :::
# % Edit: 

# %% tags=["remove-cell"]
import sinnfull
sinnfull.setup('theano')
import sinn
#sinn.config.trust_all_inputs = True  # Allow deserialization of arbitrary function

# %% tags=["remove-cell"]
import logging
logging.basicConfig()
tasklogger = logging.getLogger("smttask.smttask")
tasklogger.setLevel(logging.DEBUG)

# %% tags=["remove-input"]
if exec_environment == "notebook":
    from IPython.display import display
    import textwrap
    import holoviews as hv
    hv.extension('bokeh')

# %% tags=["remove-cell"]
import functools
import numpy as np
import pymc3 as pm
import pint

import smttask
import theano_shim as shim
from mackelab_toolbox.optimizers import Adam

# %% tags=["remove-cell"]
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    smttask.config.record = False

# %% tags=["hide-input"]
from sinnfull.models import objectives, priors, Prior
import sinnfull.models
import sinnfull.optim
from sinnfull import ureg
from sinnfull.parameters import ParameterSet, apply_sgd_masks
from sinnfull.models import OUInput, TimeAxis, Regularizer #, parameter_generators
from sinnfull.data_objects.synthetic import DataAccessor
#from sinnfull.sampling import sample_baseline_segment

from sinnfull.tasks import (CreateSyntheticDataset, CreateOptimizer, CreateModel,
                          CreateBaselineSegmentSampler, OptimizeModel)
from sinnfull.rng import get_fit_rng, get_sim_rng
from sinnfull.optim import AlternatedOptimizer, Recorder
from sinnfull.optim.convergence_tests import ConstantCost, DivergingCost
from sinnfull.data_objects.synthetic import DataAccessor as SyntheticDataAccessor
from sinnfull.utils import recursive_dict_update

from sinnfull import projectdir

# %% [markdown]
# ## Workflow parameters

# %% [markdown]
# Modifiable parameters are set in the following cell. Because we execute the notebook with Papermill, there can be only one parameters cell (identified by giving it the tag “parameters”).
#
# - To fit parameters with ground-truth latents: \
#   Move the entries in `latent_hists` to `observed_hists`.
# - The used optimizer is currently hard-coded, but one can add a flag parameter to allow choosing between different ones.

# %% tags=["parameters"]
# This cell is tagged 'parameters' for Papermill
reason = None
θ_init_key = 'ground truth'
nsteps = 5000
#fit_hyperθ_updates = {'params': {'b1': 1., 'b2': 1.}}
fit_hyperθ_updates = {}
task_save_location = 'tasklist'
step_kwargs = {}
model_rngkey = 1
optimizer_rngkey = 2
param_rngkey = 3   # Base key: keys are generator as (param_key, i)
sim_rngkey = 4     # Base key: keys are generator as (sim_key, i)

# Possible values listed in sinnfull.models.models
model = 'Ricker'
observed_hists    =['N']
latent_hists      =[]

from sinnfull.optim import learning_params
default_learning_params = learning_params[model].default   # Defined in [projectdir]/sinnfull/optim/paramsets

# Possible values listed in sinnfull.models.objectives
default_objective = objectives[model].logp_forward
params_objective  = None  # None = use default_objective
latents_objective = None  # None = use default_objective
prior = priors[model].default(M=1)
    # NB: Different priors may have different parameters

exec_environment = "module"   # Changed to 'papermill' by sinnfull.utils.generate_task_from_nb

# %% [markdown]
# An alternative to using papermill to execute the notebook, is to use the function
#
# ```python
# sinnfull.utils.run_as_script('sinnfull.workflows.Optimize_task_template',
#                              param1=value1, ...)
# ```
#
# The places parameter values in a global dictionary (`sinnfull.utils.script_args`), which the code below retrieves.

# %%
if __name__ != "__main__":
    # Running within an import
    #  - if run through `utils.run_as_script`, there will be parameters in `utils.script_args`
    #    which should replace the current values
    from sinnfull.utils import script_args
    if __name__ in script_args:
        g = globals()
        # One of the script_args will set exec_environment = "script"
        for k, v in script_args[__name__].items():
            if k in g:
                g[k] = v

# %%
if __name__ == "__main__" and exec_environment == "module":
    exec_environment = "notebook"

# %% [markdown]
# **Conversion of papermill string arguments to objects**
# To avoid dealing with serialization, the papermill arguments which should be Python objects are instead passed a string matching their identifier. For models, this works because subclasses of `~sinn.models.Model` register themselves to `mackelab_toolbox.iotools`.

# %%
# These imports only used for parameter conversion
import mackelab_toolbox as mtb
import mackelab_toolbox as iotools
import sinnfull.models
from sinnfull.models.objective_types import AccumulatedObjectiveFunction
from ast import literal_eval as make_tuple

if isinstance(model, str):
    model_name = model
    ModelClass = mtb.iotools._load_types[model_name]
else:
    ModelClass = type(model)
    model_name = ModelClass.__name__
if θ_init_key[0] == '(':
    θ_init_key = make_tuple(θ_init_key)
if isinstance(default_objective, str):
    default_objective = AccumulatedObjectiveFunction.parse_raw(default_objective)
if isinstance(nodyn_objective, str):
    nodyn_objective = AccumulatedObjectiveFunction.parse_raw(nodyn_objective)
prior = Prior.validate(prior)

# %% [markdown]
# ## Load hyperparameters
# Start by loading the file with defaults, then replace those values given in `fit_hyperθ_updates`.

# %%
fit_hyperparams = ParameterSet(default_learning_params,
                               basepath=sinnfull.optim.paramsets.basepath)
recursive_dict_update(fit_hyperparams, fit_hyperθ_updates, allow_new_keys=True)  # Notebook parameter
fit_units = fit_hyperparams.units
fit_hyperparams.remove_units(fit_units)

# %%
T    = fit_hyperparams.T
Δt   = fit_hyperparams.Δt
time = TimeAxis(min=0, max=T, step=Δt, unit=fit_units['[time]'])

# %% [markdown]
# ## Define the synthetic data

# %% [markdown]
# The synthetic data accessor replaces actual data with model simulations. Thus it needs a _model_, as well as a _prior_ on the model's parameters. The prior is sampled to generate parameter sets. Here we use the same prior as for the inference, although this need not be the case.
#
# There should be as many `param_keys` as `sim_keys`. One may repeat `param_keys`, to simulate multiple trials with the same parameters. Duplicate `sim_keys` are also permitted, although a good use case is unclear.
#
# Note that `CreateSyntheticDataset` expects a model name rather than an instance; this is to avoid having to create throwaway variables just to instantiate a model. The model class is retrieved in the same way as `ModelClass` above.

# %%
## Instantiate data accessor ##
data = CreateSyntheticDataset(
    projectdir=projectdir,
    model_name=model_name,
    time      =time,
    prior     =prior,
    param_keys=[(param_rngkey,i) for i in range(1)],
    sim_keys  =[(sim_rngkey,i) for i in range(1)]
)

# %% [markdown]
# The `SegmentSampler` creates an infinite iterator which provides a new segment on every call.
# The `trial_filter` argument is passed to `data.sel(...)`, and allows to restrict segments to
# certain trials, or certain time windows.

# %%
segment_iterator = CreateBaselineSegmentSampler(
    data=data, trial_filter={},
    t0=0*ureg.s, T=fit_hyperparams.T*ureg.s, rng_key=(5,))

# %% [markdown]
# It's a very good idea to have a look at the synthetic data before committing CPU time to fitting it. Once you are confident the generated data is as expected, replace `True` by `False` to avoid plotting it unnecessarily.

# %%
if True and exec_environment == "notebook":
    seg_iter = segment_iterator.run()
    sampled_segment = next(iter(seg_iter))[2]

    # Currently the formatting & line breaks are ignored, but according to the docs they should
    # So leaving this as-is, and when holoviews fixes their bug it'll look nicer (https://github.com/holoviz/holoviews/issues/4743)
    lines = []
    for name, value in seg_iter.data.trials.trial.data[0].params:
        lines.append(f'<b>{name}</b><br>')
        if isinstance(value, np.ndarray):
            lines.append(textwrap.indent(np.array_repr(value, precision=4, suppress_small=True),
                                         "&nbsp;&nbsp;")
                         .replace("\n","<br>\n")+"<br><br>")
        else:
            lines.append(textwrap.indent(str(value), "&nbsp;&nbsp;")
                         .replace("\n","<br>\n")+"<br><br>")
    param_vals = hv.Div("\n".join(lines))

    curves = [hv.Curve(data_var.data[:,i],
                       kdims=list(data_var.coords), vdims=[f"{data_var_name}_{i}"])
              for data_var_name, data_var in sampled_segment.data_vars.items()
              for i in range(data_var.shape[1])]
    data_panel = hv.Layout(curves).opts(hv.opts.Curve(height=150, width=200))

    # FIXME: It should be possible to make a nested layout, so params appear
    #        to the side, but I can't figure out how
    display((data_panel.cols(2) + param_vals).cols(2).opts(title="First training sample"))

# %% [markdown]
# ## Set the model parameters
#
# Valid options:
# - integer tuple: Used as a key to sample initialization parameters from `prior`.
# - `'ground truth'`: Start fit from the ground truth parameters.
# - file name: Load parameters from provided file.

# %%
valid_options = ['ground truth', 'file', 'test']
if isinstance(θ_init_key, tuple):
    θ_init = prior.random(θ_init_key, space='optim')
elif θ_init_key == 'ground truth':
    θ_init = prior.random((param_rngkey,0), space='optim')
elif os.exists(θ_init_key):
    # Untested
    θ_init = ParameterSet(θ_init_key)
else:
    raise ValueError(f"Unrecognized value for `θ_init_key`: {θ_init_key} (type: {type(θ_init_key)}\n"
                     f"It should be either an RNG key (int tuple), or one of {valid_options}")

# %% [markdown]
# Setting the initialization with ground truth works because it is equivalent to
#
# - draw values directly in the model space;
# - or draw them in the optim space and backtransform them to the model
#
# (in both cases, PyMC3 draws the optim variables and transforms them). One can check this with the following::
#
# ```python
# model_θ = prior.random((0,))
# backtransformed_θ = prior.backward_transform_params(
#     prior.random((0,), space='optim'))
# assert all(np.all(np.isclose(model_θ[nm], backtransformed_θ[nm]))
#            for nm in model_θ)
# ```

# %% [markdown]
# Store numeric values, so
# - future changes don't change stored initial values;
# - it doesn't matter _how_ we generated the initialization parameters, just what their values are.

# %%
if hasattr(θ_init, 'get_values'):
    θ_init = θ_init.get_values()

# %% [markdown]
# ## Instantiate the model
#
# Having chosen initial parameters `θ_init`, instantiate the model. The `rng_key` seeds the RNG used to integrate the model.

# %%
modelθ_init = prior.backward_transform_params(θ_init)
model = CreateModel(model_class=model_name, time=time,
                    params=modelθ_init, rng_key=(model_rngkey,))

# %% [markdown]
# ## Define the optimizer

# %% [markdown]
# ### Objective functions
#
# Objective functions are model-specific and defined in [*sinnfull.models*](../models). They can be retrieved by name from the `objectives` dictionary.
#
# The optimizer has a multiple keyword arguments for objective functions, to allow specifying different objectives for the parameters vs latents, or for the edges of the data segment.
#
# - `logp`          : Default objective for both parameters and latents
# - `logp_params`   : Default objective for the parameters
# - `logp_latents`  : Default objective for the latents
# - `prior_params`  : A PyMC3 model used as prior. Since they are mathematically equivalent, this can also be used to provide a regularizer.
# - `prior_latents` : Currently not used.
#
# It suffices to specify `logp` to get a valid optimizer; alternatively, `logp_params` and `logp_latents` may be specified together. In case of redundantly specified objectives, the more specific takes precedence.
#
# > **NOTE** `prior_params` should be an instance of [`Prior`](../models/objective_types.py), which itself subclasses `PyMC3.Model`.
#
# > **NOTE** `prior_params` is used for two things:
# >
# > - Providing a prior / regularizer on the parameters.
# > - Specifying which parameters will be optimized.
# >
# > Specifically, the model parameters the optimizer will attempt to fit are exactly those returned by `prior_params.optim_vars`.
#
# The name _“logp”_ is borrowed from the name of the analogous function in [PyMC3](https://docs.pymc.io/Probability_Distributions.html). In fact any scalar objective function is acceptable; it doesn't have to be a likelihood, or even a probability.

# %% [markdown]
# ### Updating hyper parameters during optimization
# `AlternatedOptimizer` provides a callback hook to update the hyper parameters on each pass. Here we use it to scale the latent learning rate by the standard deviation of the theoretical stationary distribution for the latents.
# This expects the model to define a `stationary_stats()` method, returning a dictionary of floats.
#
# The first thing we do is check that the model indeed provides this method, and that the returned dictionary is in the expected format. Since this test is run during workflow creation instead of run execution, it catches errors immediately rather than waiting for them to be created, queued and executed.

# %%
from collections.abc import Callable
import inspect
from numbers import Number
from sinn import History, Model

# %%
test_model = model.run(cache=False)
required_stats = {'std'}  # The statistics used by update_hyperθ
hist_names = {h.name for h in test_model.history_set}

if not isinstance(test_model, Model):
    raise TypeError(f"`model.run()` should return a sinn Model, but instead returned a {type(test_model)}.\n"
                    f"model_name: {model_name}\nModelClass: {ModelClass}")

if not hasattr(test_model, 'stationary_stats_eval'):
    raise ValueError(f"{test_model.name} does not provide the required 'stationary_stats_eval' method.")
elif not isinstance(test_model.stationary_stats_eval, Callable):
    raise ValueError(f"{test_model.name}.stationary_stats_eval is not callable")

stats = test_model.stationary_stats_eval()
if not isinstance(stats, dict):
    raise ValueError(f"{test_model.name}.stationary_stats must return a dictionary. Returned: {stats} (type: {type(stats)}).")
non_hist_keys = [k for k in stats if k not in hist_names]
if non_hist_keys:
    raise ValueError(f"{test_model.name}.stationary_stats must return a dictionary where keys are strings matching history names. Offending keys: {non_hist_keys}.")
#stats = {h.name: v for h,v in stats.items()}
missing_hists = set(latent_hists) - set(stats)
if missing_hists:
    raise ValueError(f"{test_model.name}.stationary_stats needs to define statistics for all latent histories. Missing: {missing_hists}.")

not_a_dict = [h_name for h_name in latent_hists if not isinstance(stats[h_name], dict)]
if not_a_dict:
    raise ValueError(f"{test_model.name}.stationary_stats[hist name] must be a dictionary. Offending entries: {not_a_dict}.")
missing_stats = [h_name for h_name in latent_hists if not required_stats <= set(stats[h_name])]
if missing_stats:
    raise ValueError(f"{test_model.name}.stationary_stats must define the following statistics: {required_stats}. "
                     f"Some or all of these are missing for the following entries: {missing_stats}.")

return_vals = {f"{h_name} - {stat}": stats[h_name][stat]
               for h_name in latent_hists for stat in required_stats}
does_not_return_number = {k: f"{v} (type: {type(v)})" for k,v in return_vals.items()
                          if not isinstance(v, (Number, np.ndarray))}
if does_not_return_number:
    raise ValueError(f"{test_model.name}.stationary_stats must return a nested dictionary of plain numbers or Numpy arrays. "
                     f"Offending entries:\n{does_not_return_number}")

del stats, non_hist_keys, missing_hists, not_a_dict, missing_stats, return_vals, does_not_return_number


# %% [markdown]
# Having validated the model's `stationary_stats()` method, we use it to define a hyperparameter update function which scales the learning rate to each history's variance.

# %%
## Hyperparams update callback ##
def update_hyperθ(optimizer):
    """
    This function is called at the beginning of each `step` to update the hyper parameters.

    .. Note:: This function's code is serialized into the task description.
       Therefore it must not depend on any global variables or functions.

    :returns: A nested dictionary matching the structure of `fit_hyperparams`. Not all
        entries are required; those provided will replace the ones in `fit_hyperparams`.
    """
    λη = optimizer.orig_fit_hyperparams['latents']['λη']
    stats = optimizer.model.stationary_stats_eval()
    updates = {'latents': {'λη': {}}}
    for h in optimizer.latent_hists:
        updates['latents']['λη'][h.name] = λη*stats[h.name]['std']
    return updates


# %% [markdown]
# #### Early-stopping conditions
#
# We add two tests for stopping a fit early:
#
# - `constant_cost` will return `Converged` if the last 4 recorded $\log L$ values are all within 0.3 of each other.<br>
#   (Equiv to: for any given pair, one parameter set is at most 35% ($= 1 - e^{0.3}$) more likely than the other.)
# - `diverging_cost` will return `Failed` if the current cost value is lower than the initial value, or NaN.<br>
#   This is based on the observation that randomly initialized fits almost always start with a sharp increase in the $\log L$. If further iterations annul this initial progress, the fit does not seem able to make stable progress.
#   > Depending on circumstance, `diverging_cost` may be too strict for fits starting from ground truth parameters, since those start with a relatively high likelihood.

# %%
constant_cost = ConstantCost(cost_recorder='log L', tol=0.3, n=4)
diverging_cost = DivergingCost(cost_recorder='log L', maximize=True)

# %% [markdown]
# #### Create the optimizer
# The last step is to assemble everything into an optimizer.

# %%
## Instantiate the optimizer ##
optimizer = CreateOptimizer(
    model                  =model,
    rng_key                =(optimizer_rngkey,),
    data_segments          =segment_iterator,
    observed_hists         =observed_hists,
    latent_hists           =latent_hists,
    prior_params           =prior,
    init_params            =θ_init,
    fit_hyperparams        =fit_hyperparams,
    update_hyperparams     =update_hyperθ,
    logp                   =default_objective,
    logp_params            =params_objective,
    logp_latents           =latents_objective,
    convergence_tests      =[constant_cost, diverging_cost]
)

# %% [markdown]
# ## Create the recorders
#
# Recorders are attached to the optimizer by the `Optimize` task, and record the optimizer's state during the optimization process. The recording frequency can be set independently for each recorder (so e.g. the expensive `latents_recorder` is executed less often).
#
# > Recorders are defined in [_optim/recorders_](../optim/recorders.py).

# %%
from sinnfull.optim.recorders import (
    LogpRecorder, ΘRecorder, LatentsRecorder)

# %%
logL_recorder = LogpRecorder()
Θ_recorder = ΘRecorder(keys=tuple(prior.optim_vars.keys()))
latents_recorder = LatentsRecorder(optimizer)

# %% [markdown]
# ## Optimization task

# %% [markdown]
# The final task, `OptimizeModel`, is the only recorded one (all the others are `@MemoizedTask`'s). Its definition is quite simple:
#
#   - Attach the recorders to the optimizer.
#   - Iterate the optimizer for the number of steps specified by `nsteps`.
#
# Because it is a `@RecordedIterativeTask`, it is able to continue a fit from a previous one with fewer steps. *Iterative* tasks define two additional attributes compared to recorded ones:
#
#   - An *iteration parameter* (in this case, `nsteps`). This must be an integer, must be among both the inputs and outputs, and must increase by 1 for each iteration.
#   - An *iteration map*: A dictionary mapping outputs of one iteration to the inputs of the next. In this case the map is simply `{'optimizer: 'optimizer', 'recorders': 'recorders'}`.

# %%
#for optimizer, nsteps in zip(optimizers, nsteps_list):
if reason is None:
    reason = "Hyperparameter exploration\n"
reason += \
f"""
- OU + Linear projection
- Synthetic data
- Init params: {θ_init_key}
- {nsteps} passes
""".rstrip()
for k, v in fit_hyperθ_updates.items():
    if isinstance(v, pint.Quantity):
        reason += f"\n- {k}: {v:~}"
    else:
        reason += f"\n- {k}: {v}"

optimize = OptimizeModel(reason   =reason,
                         nsteps   =nsteps,
                         optimizer=optimizer,
                         recorders=[logL_recorder, Θ_recorder, latents_recorder],
                         step_kwargs=step_kwargs
                        )

# %% [markdown]
# ## Export or run the task
#
# At this point, we either
#
# - Save the created task to a file, so it can be run later.
#   (If this code is being executed as a script.)
# - Execute the task.
#   (If this code is being executed in a notebook.)

# %%
if exec_environment != "notebook":
    # Notebook is either being run as a script or through papermill
    optimize.save(task_save_location)

# %% [markdown]
# We can visualize the workflow by calling the `draw` method of the highest level task. It's a quick 'n dirty visualization, but still sometimes a useful way to see dependencies.
#
# ::::{margin}
# :::{hint}  
# The Task nodes in this diagram correspond to the hexagonal nodes in the [flowchart](optimize-wf-flowchart) at the top.  
# :::
# ::::

# %%
if exec_environment == "notebook":
    optimize.draw()

# %% [markdown]
# The call to `run()` will recurse through the workflow tree, executing all required tasks. Since `OptimizeModel` is a `RecordedTask`, its result is saved to disk so that it will not need to be run again in the future if called with the same parameters.
# (To force a rerun, e.g. if the model code changed, one can execute `optimize.run(recompute=True)`.)

# %%
if exec_environment == "notebook":
    result = optimize.run(record=False, recompute=True)

# %% [markdown]
# ---

# %%
