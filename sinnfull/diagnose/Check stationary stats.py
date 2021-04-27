# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (sinnfull)
#     language: python
#     name: sinnfull
# ---

# %% [markdown]
# # Check stationary stats
#
# If a model defines stationary statistics, these can be used to verify the correctness of its update equations. One simply integrates the model, and compares the result of the analytical expressions to averages over time points.
#
# The most common is to test the model definition by importing the appropriate class from _sinnfull.models_ and using its *test_parameters()* to create a new model with random parameters.
#
# Alternatively, one can recreate the model used for a particular fit. The model's *update_params()* method can be used to set the model's parameters to a particular point of the fit.
#
# > **On whether to setup with `'numpy'` or `'theano'`:** \
# > A `'theano'` model has a slower instantiation but integrates much faster, so this is convenient for computing the many data points required for stationary distributions. \
# > On the other hand, if you want to probe the model (e.g. by placing a `breakpoint()` within one of
# > its update functions), than `'numpy'` is unquestionably the best choice. Debugging Theano functions is notoriously unpleasant. \
# > Note that _sinnfull_ can only be setup once, so to change between numpy and Theano, it is necessary to reload the notebook.

# %%
import sinnfull
sinnfull.setup('theano', view_only=True)

# %%
import logging
import holoviews as hv
from sinnfull.viewing import BokehOpts
from sinnfull.models import models, TimeAxis

# %%
import numpy as np
import pandas as pd
import theano_shim as shim

# %%
logger = logging.getLogger(__name__)
bokeh_opts = BokehOpts()

# %% [markdown]
# Choose which model to test.
#
# > When selecting no. of time points and step size, ensure you have a long enough trace to perform statistics. Also ensure the step is is small enough for the integration to match the analytics.

# %%
time = TimeAxis(min=0, max=1000, step=2**-6)  # Use shorter times with NumPy
model = models.OUInput4(time=time, params=models.OUInput4.get_test_parameters(),
                       rng=shim.config.RandomStream())

# %% [markdown]
#     Θ = models.OUInput.Parameters(μtilde=[0.], τtilde=[1.], σtilde=[1.], Wtilde=[[1.]], M=1, Mtilde=1)
#     model = models.OUInput(time=time, params=Θ, rng=shim.config.RandomStream())

# %%
model.params.M, model.params.Mtilde

# %% [markdown]
#     # τtilde and σtilde will affect the amount of data required to estimate stationary statistics
#     Θ = model.params.get_values()
#     Θ.τtilde, Θ.σtilde

# %% [markdown]
# Alternatively, one can also retrieve a model from a particular fit. This might given a different than giving the model directly if there is a bug in the construction of the fit task.
# ```python
# from sinnfull.viewing import RSView
# from sinnfull.viewing.record_store_viewer import FitData
# rsview = RSView()
# record = rsview.get("20210325-033910_1727a0")
# fit = FitData(record=record)
# model = fit.model
# model.update_params(fit.Θ_evol[0])
#     # Set model parameters to a particular fit step
# ```

# %% [markdown]
# Ground truth parameters for the fit can be obtained as follows:
# ```python
# fit.ground_truth_Θ()
# ```

# %% [markdown]
# Integrate the model to fill the histories with data.

# %%
model.integrate(upto='end')

# %% [markdown]
# Define NumPy equivalents for each analytical statistic; they should take as input the result of ``hist[T:]``. Axis ``0`` is always the time dimension.

# %%
stat_functions = {
    'avg': lambda data: np.mean(data, axis=0),
    'std': lambda data: np.std(data, axis=0, ddof=1)
}

# %% [markdown]
# Set a time interval for discarding initial transients.
# > Use a `float` to define the interval in units of time, or an `int` to define it in time bins.

# %%
T = 10.

# %% [markdown]
# Display stationary statistics computed with the full time trace.

# %%
all_stats = model.stationary_stats()
for hist, stats in all_stats.items():
    print(hist.name); print("-"*30)
    for stat, statvals in stats.items():
        print(" "*(len(str(hist.shape))+1), f"{stat} (analytic)   {stat} (empirical)")
        emp_stat = stat_functions[stat](hist[T:].eval())
        for idx in np.ndindex(hist.shape):
            print(f"{idx} {' '*len(stat)} {statvals[idx]:>7.4f}   {' '*len(stat)}    {emp_stat[idx]:>7.4f}")

# %% [markdown]
# Plotting the histories is useful to check that the processes really are stationary, and that the initial exclusion window is long enough.

# %%
panels = []
for hist in all_stats:
    # Add one curve per history component
    p = hv.Overlay([hv.Curve(zip(hist.time_stops, hist.get_trace(idx)),
                             kdims=['t'], vdims=[f"{hist.name}_[{str(idx).strip('(),')}]"],
                             label=f"{hist.name}[{str(idx).strip('()')}]")
                    for idx in np.ndindex(hist.shape)],
                   label=hist.name)
    p.opts(hv.opts.Curve(width=500))
    # Add shaded region indicating discard window
    w = hv.VSpan(hist.t0, hist.time[T],
                 kdims=['t', hist.name], label=hist.name,
                 )
    w.opts(color='#CCCCCC')
    wp = w*p
    wp.opts(legend_position='right')
    panels.append(wp)
hv.Layout(panels, label=hist.name).cols(1)

# %% [markdown]
# ~Finally, time evolutions of the statistics (cumputed on the interval $[T, t]$ for $t \in [t_0 + 2(T-t_0), t_n]$) give us the best sense of whether we have enough data to make an analytical comparison.~
# We can get a good sense of the value the empirical stats converge to (and therefore if it is comparable to the analytic value) by computing the statistics on disjoint windows of the data; this helps avoid correlations between statistics, which tend to bias the apparent limit. We do this for different window widths, such that one can evaluate the convergence visually by moving the slider (note in particular that for small windows, it is expected for variance to be underestimated due to the correlation between points.)

# %%
layouts = {}
for width in [20, 100, 500, 1000, 2000, 4000, 8000]:
    panels = []
    color_cycle = hv.Cycle.default_cycles['default_colors']
    for hist, stats in all_stats.items():
        #tstops = np.linspace(hist.t0+2*(hist.time[T]-hist.t0), hist.tn)
        data = hist[T:].eval()
        for stat, statvals in stats.items():
            tstops = [T+k*hist.time.step for k in range(0, len(data)-width+1, width)]
            emp_vals = [stat_functions[stat](data[k:k+width])
                        for k in range(0, len(data)-width+1, width)]
            p = hv.Overlay([hv.Curve([(t+width*hist.time.step/2, v[idx]) for t, v in zip(tstops, emp_vals)],
                                     kdims=['t'], vdims=[f"{stat}_{hist.name}_[{str(idx).strip('(),')}]"],
                                     label=f"{stat} – {hist.name}[{str(idx).strip('()')}]")
                            for idx in np.ndindex(hist.shape)],
                           label=f"{stat} – {hist.name}")
            p = p.redim(t=hv.Dimension("t", range=(hist.t0, hist.tn)))
            #p.opts(hv.opts.Curve(color=color_cycle))
            w = hv.VSpan(hist.t0, hist.time[T],
                 kdims=['t', hist.name], label=hist.name,
                 )
            w.opts(color='#CCCCCC')
            p_true = hv.Overlay([hv.HLine(statvals[idx]) for idx in np.ndindex(hist.shape)])
            p_true.opts(hv.opts.HLine(alpha=.5, line_dash='dashed'))
            #p_true.opts(hv.opts.Curve(color=color_cycle))
            #p_true.opts(line_dash='dashed')
            wp = w*p_true*p
            wp.label = f"{stat} - {hist.name}"
            wp.opts(legend_position='left')
            #wp.opts(hv.opts.Curve(alpha=.5, line_dash='dashed'))
            panels.append(wp)
    layouts[width*hist.time.step] = hv.Layout(panels).opts(hv.opts.Curve(width=550)).cols(1)
hv.HoloMap(layouts, kdims=['window']).collate()

# %% [markdown]
# To obtain a new realization with different random seed, it suffices to `clear()` and `integrate()` again.
# ```python
# model.clear()
# model.integrate(upto='end')
# ```

# %%
model.clear()
model.integrate(upto='end')
