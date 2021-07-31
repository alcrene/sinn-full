# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# # One-step-forward likelihood test – Optimizer objective

# %% [markdown] tags=["remove-cell"]
# > **Note**
# > This notebook is “report-ready”.  
# > Documentation cells for users (like this one) are already tagged for exclusion by Jupyter Book, so that generated reports contain only the relevant results.

# %% [markdown]
# This is a variant of the standard [one-step-forward likelihood test](./One-step-forward%20cost) which sets model parameters and latents to reproduce those which occurred during a recorded fit. 
#
# In contrast to the standard variant, here the exact function used to compute the loss is reproduced, including any variable substitutions, combination with the prior and compilations with Theano. It is thus a test of both the correctness objective and model, and of the correctness of the optimizer. However, it is also more difficult to distinguish errors due to these different sources. Testing only partial objectives would also difficult to do without reimplementing part of the optimizer's compilation routines.
#
# The recommendation is to first run the standard variant to test correctness of objectives and model, and once those are satisfactory to run this test to also verify the optimizer.
#
# This test also allows to see how the objective evolves over the course of a fit, something which is not possible with the standard variant. In particular, one can detect if the objective drifts away from the ground-truth solution during the fit.

# %% tags=["remove-cell"]
import sinnfull
sinnfull.setup('theano', view_only=True)

# %% tags=["remove-cell"]
import itertools
import numpy as np
import pandas as pd
import theano_shim as shim
from smttask import Task
from mackelab_toolbox.utils import index_iter, GitSHA
from sinnfull.models import models, TimeAxis, objectives, ObjectiveFunction
from sinnfull.models.base import Regularizer, PureFunction
from sinnfull.diagnostics.one_step_cost import OneStepForwardLogp
from sinnfull.diagnostics.utils import set_to_step
from sinnfull.viz import pretty_names, RSView, FitData
from sinnfull.viz.utils import get_histdim_name

# %% tags=["remove-input"]
import holoviews as hv
hv.extension('bokeh')

# %% tags=["remove-cell"]
rsview = RSView()

# %%
num_test_samples = 1000

# %% [markdown]
# Load the data from fit we want to investigate.

# %% [markdown] tags=["remove-cell"]
# > Run labels can be obtained either with the [Result viewer](../view/Result%20viewer.ipynb), or with Sumatra's [web interface](https://sumatra.readthedocs.io/en/latest/web_interface.html).

# %%
record = rsview.get('20210608-184539_4efe63')

# %% tags=["remove-cell"]
optimizer = Task.from_desc(record.parameters).optimizer.run()
fitdata = FitData(record=record, Θspace='optim')

# %% [markdown] tags=["remove-cell"]
# Set the data to a particular fit step. The step number will be rounded to the nearest step at which latents were recorded.

# %% tags=["remove-cell"]
set_to_step(optimizer, fitdata, 0)


# %%
def logp(model, k):
    return optimizer.logp(k, 1)  # Look at 1 time point => batch size of 1


# %%
model = optimizer.model
latent_logp_graph, _ = optimizer.logp_latents(model.curtidx_var, model.batchsize_var)
latent_logp_f = shim.graph.compile([model.curtidx_var, model.batchsize_var],
                                   latent_logp_graph)
def logp_latents(model, k):
    return latent_logp_f(k, 1)  # Look at 1 time point => batch size of 1


# %% [markdown]
# Pick a reference point $t$ for the test. Here we choose the point half way between $t_0$ and $t_{max}$.

# %%
test_t = optimizer.model.t0 + (optimizer.model.cur_t - optimizer.model.t0)/2

# %% [markdown]
# Run the test

# %% tags=["remove-cell"]
# List format:
# - One entry for each model-objective pair we want to test
# - Each entry is composed of (model, objective, optim step, test label)
obj_tests = [(optimizer.model, logp, 0, "full logp"),
             (optimizer.model, logp, 4980, "full logp"),
             #(optimizer.model, logp_latents, 0, "latents logp"),
             #(optimizer.model, logp_latents, 4980, "latents logp")
            ]

# %% [markdown] tags=["remove-cell"]
# Store a ground truth value for each model used in a test, before the histories are cleared.

# %% tags=["remove-cell"]
_models = {id(m): m for m, *_ in obj_tests}  # Dictionary serves to remove duplicates
assert len(_models) == 1
gt = fitdata.get_observed_data().sel(time=slice(model.time.min, model.time.max))
    # Slicing ensures the t0 of the ground truth lines up with that of the model
    # (in particular, this discards any time bins reserved for padding)
ground_truth = {
    _model.name: {
        get_histdim_name(h, idx)
        : gt[hnm].data[h.get_tidx(test_t, allow_rounding=True)+1][idx]
        for hnm, h in _model.nested_histories.items() for idx in index_iter(h.shape)}
    for _model in _models.values()}

# %% [markdown] tags=["remove-cell"]
# Store the value of the objective with the ground truth value, before the histories are cleared.

# %% tags=["remove-cell"]
# FIXME: Assumes only one model
k = model.get_tidx(test_t, allow_rounding=True)
vals = ground_truth['ObservedDynamics']
ξk = np.array([vals['ξ_0'], vals['ξ_1']])
old_ξk = model.input.ξ.data[k+1]
model.input.ξ[k+1] = ξk

logp_gt = {}
for _model, obj, step, obj_name in obj_tests:
    if (_model.name, obj_name) in logp_gt:
        continue
    k = _model.get_tidx(test_t, allow_rounding=True)
    logp_gt[(_model.name, obj_name)] = obj(_model, k+1)
    
model.input.ξ[k+1] = old_ξk

# %%
frames = {}
higher_than_gt_frames = {}
for _model, obj, step, obj_name in obj_tests:
    set_to_step(optimizer, fitdata, step)
    test = OneStepForwardLogp(
        model = _model,
        logp = obj)
    test.set_t(test_t)
    test.sample_next_step(num_test_samples)
    gt = ground_truth[_model.name]
    _logp_gt = logp_gt[(_model.name, obj_name)]
    true_vline = hv.HoloMap(
        {d.name: hv.Curve([(gt[d.name], 0)], kdims=[d], label="true value")  # 1 point Curve won't plot, but will show up in the legend
                 * hv.VLine(gt[d.name], kdims=[d, 'p'])  # Annotations don't show up in legend
         for d in test.kdims},
                      kdims=['dimension']
        ).opts(hv.opts.Curve(color='#888888'),
               hv.opts.VLine(color='#888888'))
    frames[(_model.name, obj_name, step)] = \
        (test.mc_marginal_dists
         * test.p_marginals.opts(color='orange', line_width=3)
         * true_vline
        ) \
        .layout().cols(2).opts(axiswise=True) \
        .opts(hv.opts.Curve(axiswise=True),
              hv.opts.Distribution(axiswise=True),
              #hv.opts.VLine(color='#555555')
             )
    higher_than_gt = [('ground truth',
                       *(gt[d.name] for d in test.kdims),
                       _logp_gt,3)]
    higher_than_gt += \
       [(i, *s, l) 
        for i,s,l in zip(itertools.count(), test._samples, test._logps)
        if l >=  _logp_gt]
    higher_than_gt_frames[(_model.name, obj_name, step)] = \
        hv.Table(higher_than_gt, kdims=['index']+test.kdims, vdims=['log p'])

# %% tags=["remove-cell"]
hvframes = hv.HoloMap(frames, kdims=[hv.Dimension('model'),
                                     hv.Dimension('objective', label="objective function"),
                                     hv.Dimension('step', label="iteration step", type=str)]) \
           .collate() \
           .opts(hv.opts.NdLayout(framewise=True, axiswise=True),
                 hv.opts.Overlay(framewise=True, axiswise=True))

# %% [markdown] tags=["remove-cell"]
# - *full logp*: The function used to record the objection during optimization.  
#   This is compiled by the optimizer when it gets created.  
#   It is based on the objective used to optimize the parameters.
# - *latents logp*: A compiled version of the loss used to create the latents update function.  
#   If the same objective is used to optimize both parameters and latents, plotting this should be redundant with *full logp*.
#   This is compiled in this notebook, since the optimizer doesn't provide it.

# %% tags=["remove-input"]
hvframes

# %% [markdown]
# Check if there are any samples with higher likelihood than the ground truth values. While small deviations of the mode due to the prior may be expected, any moderate deviation may be indicative of a problem.

# %% tags=["remove-input"]
# Round the values for a more compact display
htgf = {}
for k, table in higher_than_gt_frames.items():
    df = table.data
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    else:
        df = df.copy()
    for c in df.columns:
        if c == 'index':
            pass
        elif c == 'log p':
            df[c] = round(df[c], 3)
        else:
            df[c] = round(df[c], 4)
    htgf[k] = hv.Table(df)

table_frames = hv.HoloMap(htgf,
                          kdims=[hv.Dimension('model'),
                                 hv.Dimension('objective', label="objective function"),
                                 hv.Dimension('step', label="iteration step", type=str)])
table_frames.opts(title="Sample points with likelihood higher than the ground truth")
table_frames

# %% [markdown]
# ## Correct latents
#
# We can check that the sequence $ξ_k$ recorded as ground truth is the same as the one used to generate the data $u_k$. Indeed, the recorded ground truth value of $ξ_{k+1}$ is (the test time index is $k+1$):

# %% [markdown] tags=["remove-cell"]
# **Note**: Everything below assumes that there is only one model, stored as `model`.

# %% tags=["remove-cell"]
from IPython.display import display_latex
from mackelab_toolbox.utils import array_to_str

# %% tags=["hide-input"]
set_to_step(optimizer, fitdata, 0)
k = model.get_tidx(test_t, allow_rounding=True)

# %% tags=["remove-input"]
vals = ground_truth['ObservedDynamics']
ξk = [vals['ξ_0'], vals['ξ_1']]
display_latex(f"$ξ_{{k+1}} = {array_to_str(ξk)}$", raw=True)

# %% [markdown]
# Which, given the history up to $k$ and the Wilson-Cowan dynamics, should produce the following value for $u_{k+1}$:

# %% tags=["remove-cell"]
model.input.ξ[k+1] = np.array(ξk)

# %% tags=["remove-input"]
uk = model.dynamics.u_upd(model.dynamics, k+1).eval()
display_latex(f"$u_{{k+1}}|u_{{k+1}}, ξ_{{k+1}} \\text{{  (computed)}} = {array_to_str(uk)}$", raw=True)

# %% [markdown]
# Which matches up with the recorded value of $u_k$:

# %% tags=["remove-input"]
uk = model.dynamics.u[k+1].eval()
display_latex(f"$u_{{k+1}} \\text{{  (recorded)}} = {array_to_str(uk)}$", raw=True)

# %% [markdown] tags=["remove-cell"]
# Same verification as above, but without relying on the model's update function.  
# Useful to understand when the verification above fails and we want to understand want.

# %% tags=["remove-cell"]
Δt = model.dt.magnitude
ukm1 = model.dynamics.u.data[k]
uk = model.dynamics.u.data[k+1]
Ik = model.dynamics.I.data[k+1]
α = model.dynamics.params.α.get_value()
β = model.dynamics.params.β.get_value()
h = model.dynamics.params.h.get_value()
w = model.dynamics.params.w.get_value()

# %% tags=["remove-cell"]
k.plain

# %% tags=["remove-cell"]
Ik

# %% tags=["remove-cell"]
model.dynamics.I.data[k+1]

# %% tags=["remove-cell"]
ukm1 + (α*Δt) * (-ukm1 + w @ (1+np.exp(-β*(ukm1-h)))**-1 + Ik)

# %% tags=["remove-cell"]
uk

# %% [markdown]
# ## Decomposition of dynamics and input objective

# %% [markdown]
# **Ground-truth $ξ_{k+1}$**
#
# We can verify that with ground truth values, the dynamical component of the loss is 0, as it should.

# %% tags=["remove-cell"]
model.input.ξ[k+1] = np.array(ξk)

# %% tags=["remove-input"]
logp_dyn   = objectives.WC(model.dynamics, k+1).eval()
logp_input = objectives.GWN(model.input, k+1).eval()
logp_total = logp_latents(model, k+1)
display_latex(f"""$$\\begin{{array}}{{ll}}
\\text{{objective (dynamics)}} & {logp_dyn:.4} \\\\
\\text{{objective (input)}}    & {logp_input:.4} \\\\ \\hline
\\text{{objective}}            & {logp_total:.4}
\\end{{array}}$$""", raw=True)

# %% [markdown]
# **Better-fitting $ξ_{k+1}$**
#
# However, there do exist points that increase the likelihood with respect to ground truth: the penalty on the dynamics is less than the gain on the latents

# %% tags=["remove-cell"]
tab = higher_than_gt_frames[('ObservedDynamics', 'full logp', 0)]
model.input.ξ[k+1] = np.array([tab['ξ_0'][1], tab['ξ_1'][1]])
    #[.8947, 9.6227])

# %% tags=["remove-input"]
logp_dyn   = objectives.WC(model.dynamics, k+1).eval()
logp_input = objectives.GWN(model.input, k+1).eval()
logp_total = logp_latents(model, k+1)
display_latex(f"""$$\\begin{{array}}{{ll}}
\\text{{objective (dynamics)}} & {logp_dyn:.4} \\\\
\\text{{objective (input)}}    & {logp_input:.4} \\\\ \\hline
\\text{{objective}}            & {logp_total:.4}
\\end{{array}}$$""", raw=True)

# %% tags=["remove-input"]
GitSHA()

# %%
