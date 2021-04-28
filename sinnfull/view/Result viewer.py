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
# # Result viewer
#
# This is a living notebook, meant as an interactive viewer for the project results. In a few locations, brackets (`{>`,`<}`) are used to indicate values that need to be set.

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
import sinnfull
sinnfull.setup('theano', view_only=True)

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
import logging
from IPython.display import display
import holoviews as hv
from holoviews.core.io import Unpickler
from sinnfull.viewing import RSView, BokehOpts, ColorEvolCurvesByMaxL
from sinnfull.utils import get_field_values

# %% tags=["remove-cell"] jupyter={"source_hidden": true}
import numpy as np
import pandas as pd

# %% tags=["remove-input"] jupyter={"source_hidden": true}
logger = logging.getLogger(__name__)
bokeh_opts = BokehOpts()
bokeh_opts.hist_records = bokeh_opts.hist_records(height=300)

# %% [markdown]
# ## Record overview

# %% [markdown] tags=["remove-cell"]
# :::{hint}
# Filtering with tags is _much_ faster than by date or label [^1].
# :::
# [^1]: This is because the tag filter is specially coded to bypass Sumatra and filter directly with the Django API.
#
# ```python
# rsview = RSView().filter.tags('finished') \
#          .filter.after({>20201223<}).filter.before({>20201231<}) \
#          .list
# rsview.add_tag('{>TAG NAME<}')
# ```

# %% tags=["remove-input"]
rsview = RSView().filter.tags('__finished__') \
          .filter.after(2021,4,12).filter.before(2021,4,15) \
          .list

# %% [markdown] tags=["remove-cell"]
#     rsview = RSView().filter.tags('{>TAG NAME<}').list

# %% tags=["remove-input"]
rsview.splitby();

# %% [markdown]
# :::{hint}
# In the figures below, click the legends to hide portions of the plot, and use the zoom & pan tools to zoom in to portions of it.
# :::

# %% tags=["remove-input"]
rsview

# %% tags=["remove-input"]
display(rsview.run_counts_df)
print("Initialization keys:")
rsview.print_used_init_keys()

# %% [markdown]
# ## Parameter set comparison
#
# Sometimes ordering the fits by achieved $\log L$ can help identify patterns – sensitivity or insensitivity to certain hyperparameters, correlation of $\log L$ with presence of NaNs, …

# %% tags=["remove-input"]
rsview.run_results().opts(editable=True)  # Editable allows selecting & copying record labels

# %% [markdown] tags=["remove-cell"]
#     # TODO: Include in table above
#     for k, split in rsview.split_rsviews.items():
#         print(k)
#         print('--------------------------')
#         print('No.rec.steps Last.rec.step')
#         for fd in split.fitcoll.values():
#             print(f"{len(fd.logL_evol):<12} {fd.logL_evol.steps[-1]}")

# %% [markdown] tags=["remove-cell"]
#     # TODO?: Include as icon in table above
#     for k, split in rsview.split_rsviews.items():
#         print(k)
#         print('--------------------------')
#         for rec in split:
#             print(rec.outcome)

# %% [markdown]
# ## Fit dynamics
#
# The ensembles of fits for each model+hyperparameters condition (>NUM<} fits per condition). In each ensemble, we find the best fitting and colour it red. Any fits which are nearly as good as this best fit are also coloured red; the more salient the red, the closer it is to this best fit.
#
# :::{dropdown} To be more precise:
# - “Best fit” is determined by looking at the window of the last 1000 fit steps. Traces are supersampled uniformly (to ensure uniform statistics), then the .3 quantile is taken as the representative $\log L$ value. The window ensures that initial transients are ignored, and the quantile provides a preference for stable, high-likelihood fits, rather than those which reach high likelihood values sporadically.
# - Colour is determined according to the following rule, where $α := \max_{\text{fits}} \log L$:
#   + $\log L = α$ → red
#   + $\log L \leq α - 2$ → grey
#   + $α - 2 < \log L < α$ → linear interpolation between red and grey with $x = \frac{α - \log L}{2}$.
#
# :::

# %% [markdown]
# :::{margin}
# When selecting different experimental conditions for these plots, remember to match the hyper parameter set ($\Lambda$) to the model; you can see which combinations are valid from the table above.
# :::

# %% tags=["remove-input"]
color_fn = ColorEvolCurvesByMaxL(rsview.logL_curves(filter='nofail'), quantile=.3, window=1000)

# %% tags=["remove-input"]
logLcurves = rsview.logL_curves(color=color_fn)
logLcurves.opts(width=300, ylim=(-12000,4500))\
          .layout(['Λ']).cols(2)

# %% tags=["remove-cell"]
## WIP: Rescale axis individually
#for p in logLcurves:
#    ymax = max(curve.data[:,1].max() for curve in p)
#    p.redim(y=hv.Dimension('log L', range=(-12000,ymax)))

# %% tags=["remove-cell"]
# Use the `exclude` argument to avoid plotting fixed parameters
θ_curves = rsview.θ_curves(exclude={'M', 'Mtilde', 'A'}, color=color_fn,
                           ground_truth=True, dynamic=False)

# %% tags=["remove-input"]
θ_curves

# %% [markdown]
# ### Ground truth fit

# %% tags=["remove-input"]
logLcurves.select(init_key='ground truth') \
    .opts(width=300, ylim=(-12000,4500))\
    .layout(['Λ']).cols(2)

# %% tags=["remove-input"]
θ_curves.select(init_key='ground truth')

# %% [markdown]
# ## Evolution of the inferred latents
#
# In the figures below, the shaded region is the ground truth data; lines are the estimate of the latent at that time step.
#
# ::::{margin}
# The lines for observed variables are the prediction of the inferred model, given the current estimate of the latents and parameters.
#
# :::{caution}
# The figure will not refresh if a non-existing combination is selected. If it seems that the frames are frozen, select another parameter combination.
# :::
# ::::

# %% tags=["remove-input"]
from sinnfull.viewing.utils import get_logL_quantile, convert_dynmap_layout_to_holomap_layout

# %% tags=["remove-cell"]
#default_split = ('Ĩ',)  # Match with `rsview.SplitKey.kdims`
default_split = rsview.SplitKey()

filtered_rsview = rsview.split_rsviews[default_split]#.filter.outcome_not("<OptimizerStatus.Failed>").list

# Choose the best fit to show as default
fitcoll = filtered_rsview.fitcoll
fits = {get_logL_quantile(fit.logL_curve.data, quantile=0.3, window=1000): fit
        for fit in fitcoll if "<OptimizerStatus.Failed>" not in fit.record.outcome}
best_fit = fits[max(fits)]

η_curves = filtered_rsview.η_curves(best_fit).cols(2)
η_curves  # Dynamic map; use for exploration

# %% [markdown]
# Once we've identified interesting values in η_curves, we select them for a static exportable map

# %% [markdown] tags=["remove-input"]
# η_curves_static = convert_dynmap_layout_to_holomap_layout(
#     η_curves,
#     filtered_rsview.all_η_step_keys,
#     # HACK: If the above fails because splitkey is not split, do the following instead:
#     # [k[1:] for k in filtered_rsview.all_η_step_keys],
#     include = dict(
#         init_key = {'ground truth', '(5, 21)', '(5, 10)', '(5, 2)', '(5, 12)'},
#         #step={0, 1, 150, 672, 1830, 5000}
#         step = filtered_rsview.all_η_steps
#     )
#
# )
# η_curves_static.redim.default(init_key='ground truth', step=5000)
# #η_curves_static

# %% [markdown] tags=["remove-cell"]
# ## Continuing runs
# (WIP: Currently the serialization of PyMC3 models is not idempotent, so the tasks created by the code below will not be able to start from the previous run.)

# %% [markdown] tags=["remove-cell"]
# from smttask import Task
# from sinnfull.viewing.record_store_viewer import FitData, StrTuple
# FitKey = FitData.FitKey

# %% [markdown] tags=["remove-cell"]
# runs_to_continue = [
#     FitKey(Λ='Λ1', init_key=StrTuple(5,25))
# ]

# %% [markdown] tags=["remove-cell"]
# for fitdata in rsview.fitcoll:
#     if True:#fitdata.key in runs_to_continue:
#         task = Task.from_desc(fitdata.record.parameters)
#         task.save("tasklist")

# %%
