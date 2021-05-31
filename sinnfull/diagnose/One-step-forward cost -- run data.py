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
#
# This is a variant of the standard [one-step-forward likelihood test](./One-step-forward%20cost) which sets model parameters and latents to reproduce those which occurred during a recorded fit. 
#
# In contrast to the standard variant, here the exact function used to compute the loss is reproduced, including any variable substitutions, combination with the prior and compilations with Theano. It is thus a test of both the correctness objective and model, and of the correctness of the optimizer. However, it is also more difficult to distinguish errors due to these different sources. Testing only partial objectives would also difficult to do without reimplementing part of the optimizer's compilation routines.
#
# Because this test uses compiled Theano functions, it takes about 30 seconds to start, but runs about 4x faster than the standard variant.
#
# The recommendation is to first run the standard variant to test correctness of objectives and model, and once those are satisfactory to run this test to also verify the optimizer.

# %%
import sinnfull
sinnfull.setup('theano', view_only=True)

# %%
import numpy as np
import itertools
import theano_shim as shim
from smttask import Task
from mackelab_toolbox.utils import index_iter, GitSHA
from sinnfull.models import models, TimeAxis, objectives, ObjectiveFunction
from sinnfull.models.base import Regularizer, PureFunction
from sinnfull.diagnostics.one_step_cost import OneStepForwardLogp
from sinnfull.diagnostics.utils import set_to_step
from sinnfull.viz import pretty_names, RSView, FitData

# %%
import holoviews as hv
hv.extension('bokeh')

# %%
rsview = RSView()

# %%
num_test_samples = 1000

# %% [markdown]
# Load the data from fit we want to investigate.
# :::{margin}
# Run labels can be obtained either with the [Result viewer](../view/Result%20viewer.ipynb), or with Sumatra's [web interface](https://sumatra.readthedocs.io/en/latest/web_interface.html).
# :::

# %%
record = rsview.get('20210601-003047_c5d58b')
optimizer = Task.from_desc(record.parameters).optimizer.run()
fitdata = FitData(record=record, θspace='optim')

# %% [markdown]
# Set the data to a particular fit step. The step number will be rounded to the nearest step at which latents were recorded.

# %%
set_to_step(optimizer, fitdata, 8217)

# %% [markdown]
# Pick a reference point $t$ for the test. Here we choose the point half way between $t_0$ and $t_{max}$.

# %%
test_t = optimizer.model.t0 + (optimizer.model.cur_t - optimizer.model.t0)/2


# %%
def logp(model, k):
    return optimizer.logp(k, 1)


# %%
model = optimizer.model
latent_logp_graph, _ = optimizer.logp_latents(model.curtidx_var, model.batchsize_var)
latent_logp_f = shim.graph.compile([model.curtidx_var, model.batchsize_var],
                                   latent_logp_graph)
def logp_latents(model, k):
    return latent_logp_f(k, 1)


# %% [markdown]
# Run the test

# %% tags=["remove-cell"]
# List format:
# - One entry for each model-objective pair we want to test
# - Each entry is composed of (model, objective, test label)
obj_tests = [#(optimizer.model, logp, "full logp"),
             (optimizer.model, logp_latents, "latents logp")]

# %% [markdown] tags=["remove-cell"]
# Store a ground truth value for each model used in a test, before the histories are cleared.

# %% tags=["remove-cell"]
# TODO: Create one function shared with kdims to create name from hist & index
_models = {id(m): m for m, _, _ in obj_tests}  # Dictionary serves to remove duplicates
ground_truth = {
    _model.name: {
        f"{pretty_names.unicode.get(h.name)}{'_' if len(idx) else ''}{''.join(str(i) for i in idx)}"
        : h.data[h.get_tidx(test_t, allow_rounding=True)+1][idx]
        for h in _model.unlocked_histories for idx in index_iter(h.shape)}
    for _model in _models.values()}

# %% [markdown]
# Store the value of the objective with the ground truth value, before the histories are cleared.

# %%
logp_gt = {}
for _model, obj, obj_name in obj_tests:
    k = model.get_tidx(test_t, allow_rounding=True)
    logp_gt[(_model.name, obj_name)] = obj(model, k+1)

# %%
frames = {}
higher_than_gt_frames = {}
for _model, obj, obj_name in obj_tests:
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
    frames[(_model.name, obj_name)] = \
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
                       *(np.round(gt[d.name], 4) for d in test.kdims),
                       np.round(_logp_gt,3))]
    higher_than_gt += \
       [(i, *np.round(s,4),np.round(l,3)) 
        for i,s,l in zip(itertools.count(), test._samples, test._logps)
        if l >=  _logp_gt]
    higher_than_gt_frames[(_model.name, obj_name)] = \
        hv.Table(higher_than_gt, kdims=['index']+test.kdims, vdims=['log p'])

# %% tags=["remove-cell"]
hvframes = hv.HoloMap(frames, kdims=[hv.Dimension('model'),
                                     hv.Dimension('objective', label="objective function")]) \
           .collate() \
           .opts(hv.opts.NdLayout(framewise=True, axiswise=True),
                 hv.opts.Overlay(framewise=True, axiswise=True))

# %% tags=["remove-input"]
hvframes

# %% [markdown]
# Check if there are any samples with higher likelihood than the ground truth values. While small deviations of the mode due to the prior may be expected, any moderate deviation may be indicative of a problem.

# %%
table_frames = hv.HoloMap(higher_than_gt_frames,
                          kdims=[hv.Dimension('model'),
                                 hv.Dimension('objective', label="objective function")])
table_frames.opts(title="Sample points with likelihood higher than the ground truth")
table_frames

# %%
GitSHA()
