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
# **Flowchart – Optimize Task workflow** Hexagonal nodes indicate steps executed as [Tasks] (../tasks/index) during workflow _execution_. Elliptical nodes are implemented within the workflow itself and executed during its _creation_.
# <a href="https://mermaid.ink/img/eyJjb2RlIjoiZmxvd2NoYXJ0IFREXG4gICAgc3ViZ3JhcGggd29ya2Zsb3dbT3B0aW1pemUgVGFzayBXb3JrZmxvd11cbiAgICBkaXNrZGF0YVtvbi1kaXNrIGRhdGFdXG4gICAgc3ludGhkYXRhW3N5bnRoZXRpYyBkYXRhXVxuICAgIGNkYXRhe3tjcmVhdGUgZGF0YSBhY2Nlc3Nvcn19XG4gICAgY3NhbXBsZXJ7e2NyZWF0ZSBkYXRhXFxuc2VnbWVudCBzYW1wbGVyfX1cbiAgICBjbW9kZWx7e2NyZWF0ZSBtb2RlbH19XG4gICAgY3ByaW9yKFtjcmVhdGUgcHJpb3JdKVxuICAgIGNpbml0KFtjaG9vc2UgaW5pdGlhbCBwYXJhbXNdKVxuICAgIGNvYmooW2Nob29zZSBvYmplY3RpdmVdKVxuICAgIGNoeXBlcihbY2hvb3NlIGh5cGVycGFyYW10ZXJzXSlcbiAgICBjb3B0aW1pemVye3tjcmVhdGUgb3B0aW1pemVyfX1cbiAgICBjcmVjW2NyZWF0ZSByZWNvcmRlcnNdXG4gICAgY3Rlc3QoW2NyZWF0ZSBjb252ZXJnZW5jZSB0ZXN0c10pXG4gICAgY29wdGltaXple3tvcHRpbWl6ZSBtb2RlbH19XG4gICAgZGlza2RhdGEgLS4tPiBjZGF0YVxuICAgIHN5bnRoZGF0YSAtLi0-IGNkYXRhXG4gICAgY3ByaW9yIC0uLT4gc3ludGhkYXRhXG4gICAgY21vZGVsIC0uLT4gc3ludGhkYXRhXG4gICAgY3ByaW9yIC0tPiBjb3B0aW1pemVyXG4gICAgY2h5cGVyIC0uLT52aHlwZXIoW3ZhbGlkYXRlIGh5cGVycGFyYW1ldGVyc10pXG4gICAgY2RhdGEgLS0-IGNzYW1wbGVyXG4gICAgY3NhbXBsZXIgJiBjaW5pdCAmIGNtb2RlbCAmIGNvYmogJiB2aHlwZXIgLS0-IGNvcHRpbWl6ZXJcbiAgICBjb3B0aW1pemVyICYgY3JlYyAmIGN0ZXN0IC0tPiBjb3B0aW1pemVcbiAgICBlbmRcblxuICAgIHN0eWxlIGNvcHRpbWl6ZSBmaWxsOiNjZGU0ZmYsIHN0cm9rZTojMTQ3ZWZmLCBzdHJva2Utd2lkdGg6MnB4XG4gICAgc3R5bGUgd29ya2Zsb3cgZm9udC13ZWlnaHQ6Ym9sZFxuIiwibWVybWFpZCI6e30sInVwZGF0ZUVkaXRvciI6ZmFsc2V9">
#     (edit)
# </a>  
# :::

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

# %% tags=["remove-cell"]
import functools
from typing import List
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
from sinnfull.parameters import ParameterSet
from sinnfull.models import TimeAxis, models, ObjectiveFunction
#from sinnfull.data.synthetic import SyntheticDataAccessor
#from sinnfull.sampling import sample_baseline_segment

from sinnfull.tasks import (CreateSyntheticDataset, CreateOptimizer, CreateModel,
                            CreateFixedSegmentSampler, OptimizeModel)
from sinnfull.rng import get_np_rng, get_shim_rng
from sinnfull.optim import AlternatedSGD, Recorder
from sinnfull.optim.convergence_tests import ConstantCost, DivergingCost
from sinnfull.utils import recursive_dict_update

from sinnfull import projectdir

# %%
from pydantic import BaseModel

# %% [markdown]
# ## Workflow parameters

# %% [markdown]
# Modifiable parameters are set in the following cell. Because we execute the notebook with Papermill, there can be only one parameters cell (identified by giving it the tag “parameters”).
#
# - To fit parameters with ground-truth latents: \
#   Move the entries in `latent_hists` to `observed_hists`.
# - The used optimizer is currently hard-coded, but one can add a flag parameter to allow choosing between different ones.
# - Models are specified with _selectors_: sets of tags that [filter](tags-taming-model-proliferation) the model collection down to exactly one model.

# %% tags=["parameters"]
# This cell is tagged 'parameters' for Papermill
reason = None
Θ_init_key = 'ground truth'
nsteps = 5000
#fit_hyperθ_updates = {'params': {'b1': 1., 'b2': 1.}}
fit_hyperθ_updates = {}
task_save_location = 'tasklist'
step_kwargs = {}
model_rngkey = 1
optimizer_rngkey = 2
param_rngkey = 3   # Base key: keys are generator as (param_key, i)
sim_rngkey = 4     # Base key: keys are generator as (sim_key, i)
sampler_rngkey = 5

# Values are tag selectors; selectors are always sets of strings.
# model_selector may be either a single select (i.e. set of tags)
# or a dictionary with entries matching submodels (+ __root__ and __connect__)
# If there are submodels, the __connect__ entry indicates how their histories
# relate. It should be a list of strings.
model_selector      = {'__root__'   : {'ObservedDynamics'},
                       'input'      : {'GaussianWhiteNoise'},
                       'dynamics'   : {'WilsonCowan'},
                       '__connect__': ['GaussianWhiteNoise.ξ -> WilsonCowan.I']}
observed_hists=['dynamics.u']  # Use dotted names to denote histories in submodels
latent_hists  =['input.ξ']

from sinnfull.optim import paramsets as optim_paramsets
default_hyperparams = optim_paramsets['WC'].default   # Defined in [projectdir]/sinnfull/optim/paramsets

# Values are tag selectors.
# Selectors are always sets of strings;
# they can be within dicts (to apply them to submodels)
# or lists (to indicate multiple objectives).
# All objectives are ultimately summed together
# (Future: we may add notation for coefficients multiplying objectives)
objective_selectors = {'input': {'GaussianWhiteNoise'},
                       'dynamics': {'WilsonCowan', 'se'}}
params_objective  = None  # None = use default_objective
latents_objective = None  # None = use default_objective
prior_spec = ParameterSet(
    {'input': {'selector': {'GWN', 'default'},
               'kwds': dict(mu_mean=[-0.25, -0.5],
                            logsigma_mean=[-1., -1.],
                            M=2)},
     'dynamics': {'selector': {'WC', 'rich'}, 'kwds': dict(M=2)}
    })
    # NB: Different priors may have different parameters

synth_param_spec = prior_spec.copy()  # Requires sinnfull.ParameterSet
synth_param_spec.update({'input.kwds.mu_std': 1.,        # Tip: Use dotted notation to avoid
                         'input.kwds.logsigma_std': .5}) # quashing other params
    
exec_environment = "module"   # Changed to 'papermill' by sinnfull.utils.generate_task_from_nb

# %% [markdown]
# **Conversion of papermill string arguments to objects**  
# To avoid dealing with serialization, the papermill arguments which should be Python objects are instead passed as strings. But this means we have to decode them.

# %%
from ast import literal_eval
if Θ_init_key[0] == '(':
    Θ_init_key = literal_eval(Θ_init_key)
g = globals()
for param in ['default_hyperparams', 'fit_hyperθ_updates',
              'synth_param_spec', 'prior_spec',
              'model_selector',  'objective_selectors']:
    pval = g[param]
    if isinstance(pval, str):
        pval = literal_eval(pval)
    elif isinstance(pval, dict):
        for k, v in pval.items():
            if isinstance(v, str):
                pval[k] = literal_eval(v)
    g[param] = pval
# JSON converts sets into lists. Convert them back to sets/tuples.
for param in ['model_selector', 'objective_selectors',
              'synth_param_spec', 'prior_spec']:
    pval = g[param]
    if isinstance(pval, dict):
        pval = ParameterSet(pval)
        for k, v in pval.flat():
            if isinstance(v, str):
                literal_eval(v)
            if 'kwds' not in k and isinstance(v, list):
                v = tuple(v)
            pval[k] = v
    g[param] = pval

# %% [markdown]
# An (experimental) alternative to using papermill to execute the notebook, is to use the function
#
# ```python
# sinnfull.utils.run_as_script('sinnfull.workflows.Optimize_task_template',
#                              param1=value1, ...)
# ```
#
# This places parameter values in a global dictionary (`sinnfull.utils.script_args`), which the code below retrieves.

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
from sinnfull.models import get_objectives, get_prior, get_model_class

# %% [markdown]
# Retrieve the model.

# %%
ModelClass = get_model_class(model_selector)

# %% [markdown]
# Retrieve all the objectives and sum them.

# %%
default_objective = sum(get_objectives(objective_selectors))

# %% [markdown]
# Retrieve the parameter distribution used to generate synthetic data. This can be the same as the prior, but if the prior is broad, it can be a good idea to sample the data parameters from tighter distributions.

# %%
synth_param_dist = get_prior(ModelClass, synth_param_spec)

# %%
if __name__ == "__main__" and exec_environment == "module":
    exec_environment = "notebook"
# Possible values of exec_environment: "notebook", "module", "papermill"

# %% tags=["remove-input"]
# Imports only used in the notebook
if exec_environment == "notebook":
    from IPython.display import display
    import textwrap
    import holoviews as hv
    hv.extension('bokeh')

# %% [markdown]
# ## Load hyperparameters
# Start by loading the file containing defaults, then replace those values given in `fit_hyperθ_updates`.

# %%
fit_hyperparams = ParameterSet(default_hyperparams,
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
# There should be as many `param_keys` as `sim_keys`. One may repeat `param_keys` (with different `sim_keys`), to simulate multiple trials with the same parameters. Duplicate `sim_keys` (with different `param_keys`) are also permitted, although a good use case is unclear.
#
# Note that `CreateSyntheticDataset` expects a model name rather than an instance; this is to avoid having to create throwaway variables just to instantiate a model. The model class is retrieved in the same way as `ModelClass` above.

# %%
synthdata_model = CreateModel(
    time              = time,
    model_selector    = model_selector,
    params            = None,  # Just pick anything to instantiate the model (Uses model.get_test_parameters().)
    rng_key           = (sim_rngkey,0)  # Just use the first key; reseeds when sampling dataset anyway
)

# %%
## Instantiate data accessor ##
data = CreateSyntheticDataset(
    projectdir=projectdir,
    model     =synthdata_model,
    #model_name=model_name,
    #time      =time,
    prior     =synth_param_dist,
    param_keys=[(param_rngkey,i) for i in range(1)],
    sim_keys  =[(sim_rngkey,i) for i in range(1)],
    init_conds = {'dynamics.u': 0}
)

# %% [markdown]
# The `SegmentSampler` creates an infinite iterator which provides a new segment on every call.
#
# - `trial_filter` argument is passed to `data.sel(...)`, and allows to restrict segments to certain trials. It can also be used to restrict time windows, but for this purpose `t0` and `T` are more convenient.
# - `t0` and `T` are used to select a fixed window to sample from.

# %%
t0 = 1*ureg.s
segment_iterator = CreateFixedSegmentSampler(
    data=data,
    trial_filter={},
    t0=t0, T=fit_hyperparams.T*ureg.s-t0,
    rng_key=(sampler_rngkey,)
)

# %% [markdown]
# It's a very good idea to have a look at the synthetic data before committing CPU time to fitting it. Once you are confident the generated data is as expected, replace `True` by `False` to avoid plotting it unnecessarily.

# %% [markdown] tags=["remove-cell"]
#     hv.renderer('bokeh').theme = 'dark_minimal'

# %%
if True and exec_environment == "notebook":
    seg_iter = segment_iterator.run()
    sampled_segment = next(iter(seg_iter))[2]

    from typing import Union
    from sinn.models import IndexableNamespace
    def get_param_div_str(params: Union[IndexableNamespace,dict]):
        # Currently the formatting & line breaks are ignored, but according to the docs they should
        # So leaving this as-is, and when holoviews fixes their bug it'll look nicer (https://github.com/holoviz/holoviews/issues/4743)
        lines = []
        if isinstance(params, dict):
            params = params.items()
        for name, value in params:
            lines.append(f'<b>{name}</b><br>')
            if isinstance(value, (IndexableNamespace, dict)):
                lines.append(textwrap.indent(get_param_div_str(value),
                                             "&nbsp;&nbsp;"))
            elif isinstance(value, np.ndarray):
                lines.append(textwrap.indent(np.array_repr(value, precision=4, suppress_small=True),
                                             "&nbsp;&nbsp;")
                             .replace("\n","<br>\n")+"<br><br>")
            else:
                lines.append(textwrap.indent(str(value), "&nbsp;&nbsp;")
                             .replace("\n","<br>\n")+"<br><br>")
        return "\n".join(lines)
    param_vals = hv.Div(get_param_div_str(seg_iter.data.trials.trial.data[0].params))

    curves = [hv.Curve(data_var.data[:,i],
                       kdims=list(data_var.coords), vdims=[f"{data_var_name}_{i}"])
              for data_var_name, data_var in sampled_segment.data_vars.items()
              for i in range(data_var.shape[1])]
    data_panel = hv.Layout(curves).opts(hv.opts.Curve(height=150, width=200))

    # FIXME: It should be possible to make a nested layout, so params appear
    #        to the side, but I can't figure out how
    display((data_panel.cols(2) + param_vals).cols(2).opts(title="First training sample"))
    
    ## Alternative (less compact but currently more readable) param value list
    # print(get_param_div_str(seg_iter.data.trials.trial.data[0].params)
    #   .replace("&nbsp;", " ")
    #   .replace("<b>", "").replace("</b>", "")
    #   .replace("<br>", "").replace("</br>", ""))

# %% [markdown]
# ## Set the model parameters
#
# Valid options:
# - integer tuple: Used as a key to sample initialization parameters from `prior`.
# - `'ground truth'`: Start fit from the ground truth parameters.
# - file name: Load parameters from provided file.

# %%
prior = get_prior(ModelClass, prior_spec)

# %%
valid_options = ['ground truth', 'file', 'test']
if isinstance(Θ_init_key, tuple):
    Θ_init = prior.random(Θ_init_key, space='optim')
    modelΘ_init = prior.backward_transform_params(Θ_init)
elif Θ_init_key == 'ground truth':
    Θ_init = synth_param_dist.random((param_rngkey,0), space='optim')
    modelΘ_init = synth_param_dist.backward_transform_params(Θ_init)
elif isinstance(Θ_init_key, set):
    # Parameter set selector
    from sinnfull.models import paramsets
    modelΘ_init = paramsets[Θ_init_key]
elif os.exists(Θ_init_key):
    # Untested
    modelΘ_init = ParameterSet(Θ_init_key)
else:
    raise ValueError(f"Unrecognized value for `Θ_init_key`: {Θ_init_key} (type: {type(Θ_init_key)}\n"
                     f"It should be either an RNG key (int tuple), or one of {valid_options}")

# %% [markdown]
# :::{note}
# :dropdown:  
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
# :::

# %% [markdown]
# Store numeric values, so
# - future changes don't change stored initial values;
# - it doesn't matter _how_ we generated the initialization parameters, just what their values are.

# %%
if hasattr(modelΘ_init, 'get_values'):
    modelΘ_init = modelΘ_init.get_values()

# %% [markdown]
# ## Instantiate the model
#
# Having chosen initial parameters `Θ_init`, instantiate the model. The `rng_key` seeds the RNG used to integrate the model.

# %%
model = CreateModel(time              = time,
                    model_selector    = model_selector,
                    params            = modelΘ_init,
                    rng_key           = (model_rngkey,))
                    #submodel_selectors= submodel_selectors,
                    #connect           = submodel_connections)

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
# > **NOTE** `prior_params` should be an instance of [`Prior`](../models/base.py), which itself subclasses `PyMC3.Model`.
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
#
# :::{note}  
# The hyperparameter update function assumes that a) there exists an `input` submodel and that b) only this `input` submodel has latent variables. Thus stationary statistics are computed for that submodel only.  
# :::

# %% [markdown]
# :::{margin} Validation
# - `CreateModel` task returns a `Model`.
# - Model defines a valid `stationary_stats` method.
# :::

    # %%
    from collections.abc import Callable
    import inspect
    from numbers import Number
    from sinn import History, Model

    test_model = model.run(cache=False)
    required_stats = {'std'}  # The statistics used by update_hyperθ
    hist_names = {h.name for h in test_model.history_set}

    test_model = test_model.input
    assert all(hname.startswith('input.') for hname in latent_hists)
    input_latent_hists = [hname.split('.',1)[1] for hname in latent_hists]
    
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
    missing_hists = set(input_latent_hists) - set(stats)
    if missing_hists:
        raise ValueError(f"{test_model.name}.stationary_stats needs to define statistics for all latent histories. Missing: {missing_hists}.")

    not_a_dict = [h_name for h_name in input_latent_hists if not isinstance(stats[h_name], dict)]
    if not_a_dict:
        raise ValueError(f"{test_model.name}.stationary_stats[hist name] must be a dictionary. Offending entries: {not_a_dict}.")
    missing_stats = [h_name for h_name in input_latent_hists if not required_stats <= set(stats[h_name])]
    if missing_stats:
        raise ValueError(f"{test_model.name}.stationary_stats must define the following statistics: {required_stats}. "
                         f"Some or all of these are missing for the following entries: {missing_stats}.")

    return_vals = {f"{h_name} - {stat}": stats[h_name][stat]
                   for h_name in input_latent_hists for stat in required_stats}
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
       
    .. Note:: This function assumes that a) there exists a submodel “input”
       and b) that this submodel is the only one with latent histories.
       
    .. Note:: The AlternatedSGD optimizer currently assumes that all latent
       histories have unique names
       (so they cannot only be distinguished by their submodel).

    :returns: A nested dictionary matching the structure of `fit_hyperparams`. Not all
        entries are required; those provided will replace the ones in `fit_hyperparams`.
    """
    λη = optimizer.orig_fit_hyperparams['latents']['λη']
    # Only the input has latent hists, therefore we only need its stationary stats
    stats = optimizer.model.input.stationary_stats_eval()
    updates = {'latents': {'λη': {}}}
    for hname in optimizer.latent_hists:
        assert hname.startswith('input.')
        updates['latents']['λη'][hname] = λη*stats[hname.split('.')[1]]['std']
    return updates


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
    init_params            =Θ_init,
    fit_hyperparams        =fit_hyperparams,
    update_hyperparams     =update_hyperθ,
    logp                   =default_objective,
    logp_params            =params_objective,
    logp_latents           =latents_objective,
    #convergence_tests      =[constant_cost, diverging_cost]
)

# %% [markdown]
# mod = model.run()

# %% [markdown]
# mod.dynamics.u.cur_tidx

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

# %% [markdown]
# ### Recorders
#
# Recorders are used by the `Optimize` task to record the optimizer's state during the optimization process. The recording frequency can be set independently for each recorder (so e.g. the expensive `latents_recorder` is executed less often).
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
# ### Early-stopping conditions
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
# ### `Optimize` Task

# %%
#for optimizer, nsteps in zip(optimizers, nsteps_list):
if reason is None:
    reason = ""
else:
    reason += "\n\n"
reason += \
f"""
- Init params: {Θ_init_key}
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
                         step_kwargs=step_kwargs,
                         recorders=[logL_recorder, Θ_recorder, latents_recorder],
                         convergence_tests      =[constant_cost, diverging_cost]
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
if True and exec_environment == "notebook":
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
# %debug

# %%
