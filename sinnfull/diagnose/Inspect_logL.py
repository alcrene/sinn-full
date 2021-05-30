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
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # Inspecting the log likelihood
#
# The log likelihood function must be redefined for each model, and therefore especially prone to issues. Issues here fall into at least two categories:
#
# **Coding errors**
#
# :   The code does not match the intended formula. These errors can be identified by computing the function on its own (outside of the optimizer) with some test values and comparing with expected values (e.g. computed by hand).
#
# **Unstable optimization**
#
# :   The intended formula may not lead to stable optimization. Here investigations mostly rely on clamping all but 1 or 2 parameters, and then doing some combination of the following:
#   - Plotting the likelihood over those parameters. Non-convex profiles may indicate the need for a change in the cost function, or for adding a regularization term. Such profiles however may highly depend on the clamped parameters, making these diagnostics an inexact science.
#   - Test fitting stability with the standard *SciPy* optimizers.
#
# ```{tableofcontents}
# ```
#
# > **NOTE** Some of these tests are dated, and their objective better accomplished by another. In particular, the test in [One-step-forward cost](./One-step-forward cost.ipynb) overlaps with some of those below.

# %%
import sinnfull
sinnfull.setup('theano')
import sinnfull.optim
sinnfull.optim.diagnostic_hooks = True
import sinn
sinn.config.trust_all_inputs = True  # Allow deserialization of arbitrary function
import smttask
smttask.config.record = False
smttask.config.load_project(sinnfull.projectdir)

# %%
import logging
logging.basicConfig()

# %%
import numpy as np
import theano_shim as shim
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
sns.set(rc={'font.sans-serif': ['CMU Bright', 'sans-serif'],
            'font.serif': ['Computer Modern Roman', 'serif'],
            'savefig.pad_inches': 0,                          # Don't change size when saving
            'figure.constrained_layout.use': True,            # Make sure there is space for labels
            'figure.figsize': [5.0, 3.0]
           })

import bokeh
import bokeh.io

# %% [markdown]
# %matplotlib widget

# %%
bokeh.io.output_notebook()

# %%
from sinnfull.diagnostics.utils import load_record, set_to_step, set_to_zero

# %%
# Load the record we want to use for our test
#rec_data = load_record('20200916-162115_4e31')
rec_data = load_record('20200924-235015_a0fa')
# Initialize params & hists with ultimate fit values
set_to_step(-1)

# %%
Itilde_T_evol = [Itilde[-1] for Itilde in rec_data.latents.Itilde]
Itilde_Tm1_evol = [Itilde[-2] for Itilde in rec_data.latents.Itilde]

# %%
model = rec_data.model
optimizer = rec_data.optimizer
latents = rec_data.latents
logp = rec_data.logp
Θ = rec_data.Θ
λη = rec_data.λη
data = rec_data.data

# %%
Θ.keys

# %% [markdown]
# ---
# ## Model log likelihood
# (copied from the task definition)

# %%
## Model log-likelihood ##
import builtins
from sinnfull import shim
def logp(self, k, print=False):
    Δt = self.dt
    Δt = getattr(Δt, 'magnitude', Δt)
    μtilde=self.μtilde; σtilde=self.σtilde; τtilde=self.τtilde
    Wtilde=self.Wtilde
    Itilde=self.Itilde; I=self.I
    norm_Ik = shim.log(σtilde*shim.sqrt(2*Δt/τtilde)).sum()
    gauss_Ik = ((Itilde(k) - Itilde(k-1) - (Itilde(k-1)-μtilde)*Δt/τtilde)**2
                / (4*σtilde**2*Δt/τtilde)
               ).sum()
    sqrerr = ((shim.dot(Wtilde, Itilde(k)) - I(k))**2).sum()
    norm_Ik = shim.eval(norm_Ik, max_cost=None)
    gauss_Ik = shim.eval(gauss_Ik, max_cost=None)
    sqrerr = shim.eval(sqrerr, max_cost=None)
    if print:
        builtins.print("  norm_Ik: ", -norm_Ik)
        builtins.print("  gauss_Ik: ", -gauss_Ik)
        builtins.print("  sqrerr: ", -sqrerr)
    return - norm_Ik - gauss_Ik - sqrerr


# %%
def cost_1t(Itilde_T):
    "'1t' stands for '1 time point'"
    model.Itilde[model.tnidx] = Itilde_T
    return -logp(model, model.tnidx, print=False)


# %%
def cost_2t(Itilde_Tm1_T): # Must flatten into 1D vector
    "'1t' stands for '2 time points'"
    model.Itilde[model.tnidx-1:] = Itilde_Tm1_T.reshape(2,-1)
    return -logp(model, model.tnidx, print=False)


# %% [markdown]
# ----
# ## Compute log L at different fit steps
# I.e., does the optimizer's likelihood match what I think it should be ?

# %%
set_to_step(-1)

# %% [markdown]
# (`T` refers to the latest time point; `Tm1` the second-to-last, i.e. $T$ *minus* 1.)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10,3))
ax = axes[0]
ax.set_title("The latent variable $\\tilde{I}$ at the end of the fit.")
ax.plot(*model.Itilde.trace);
ax.set_xlabel("time"); ax.set_ylabel("$\\tilde{I}$");
ax = axes[1]
ax.set_title("The data were trying to fit (supposedly a linear mixing of $\\tilde{I}$ )")
ax.plot(*model.I.trace);
ax.set_xlabel("time"); ax.set_ylabel("$I$");

# %% [markdown]
# The value of $\tilde{I}_T$ during the optimization process. We are going to use the last few values for our test.

# %%
fig, axes = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2,1]}, figsize=(7.5, 3))
ax = axes[0]
ax.plot(Itilde_T_evol);
ax.set_xlabel("step index (log-ish scale)"); ax.set_ylabel("$\\tilde{I}_T$");
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax = axes[1]
ax.set_axis_off()
ax.table(cellText=[[i, str(step)] for i, step in enumerate(latents.steps)],
         colLabels=["step index", "step"],
         loc="upper left"
        );
plt.show();

# %%
model = rec_data.model

# len(latents.steps) = 20
for step_idx in [15, 16, 17, 19][:4]:
    with np.printoptions(precision=4):
        I_Tm1 = np.array(Itilde_Tm1_evol[step_idx])
        I_T = np.array(Itilde_T_evol[step_idx])
        print(f"----- step = {latents.steps[step_idx]} ---")
        # Remark: Setting Tm1 invalidates T, so it must be set first
        model.Itilde[model.tnidx-1] = I_Tm1
        model.Itilde[model.tnidx] = I_T
        model.params.set_values(Θ[step_idx])
        k = model.tnidx
        print(f"Itilde[T-1]: {shim.eval(model.Itilde[k-1])}")
        print(f"Itilde[T]: {shim.eval(model.Itilde[k])}")
        for θnm, θval in Θ[step_idx].items():
            nl = '\n'
            print(f"{θnm}: {str(np.array(θval)).replace(nl, ',')}")
        print(f"log L @ {model.tn:~}: {logp(model, k, print=True)}")
        print()

# %% [markdown]
# ---
# ## In place optimization test
# Do fits converge ? Head for infinity ? Evolve chaotically ?

# %%
from scipy.optimize import minimize

# %%
set_to_step(17)

# %%
model.Itilde.unlock()

# %% [markdown]
# We start with fitting just $\tilde{I}_T$. This works as expected, and moves more or less towards $\tilde{I}_{T-1}$. \
# (There are two competing terms in the log L: the “Gaussian” term pushes $\tilde{I}_T$ towards $\tilde{I}_{T-1} - \frac{(\tilde{I}_{T-1} \tilde{μ})Δt}{\tilde{τ}}$, while the “squared error” pushes $\tilde{I}_T$ towards $\tilde{W}^{-1} I_T$)

# %%
x0 = np.array(Itilde_Tm1_evol[step_idx])
res = minimize(cost_1t, x0=x0)

# %%
res.message

# %%
μtilde=shim.eval(model.μtilde);
τtilde=shim.eval(model.τtilde);
σtilde=shim.eval(model.σtilde);
Wtilde=np.array(shim.eval(model.Wtilde))
Δt = model.dt.magnitude
Itilde = model.Itilde
k = model.tnidx

# %%
gauss_target = shim.eval(Itilde(k-1) - (Itilde(k-1)-μtilde)*Δt/τtilde, max_cost=None)
sqrerr_target = np.linalg.inv(Wtilde).dot(shim.eval(model.I(k), max_cost=None))
with np.printoptions(precision=4):
    print(f"min(log L):        {res.fun}")
    print(f"Fit result:        {res.x}")
    print("------------------------------------")
    print(f"Gaussian target:   {gauss_target}")
    print(f"Sqr. error target: {sqrerr_target}")

# %% [markdown]
# Now we see if the optimizer is still stable if we try to fit $\tilde{I}_T$ and $\tilde{I}_{T-1}$ simultaneously.

# %%
x0 = np.array([Itilde_T_evol[step_idx], Itilde_Tm1_evol[step_idx]]).flatten()
res = minimize(cost_2t, x0=x0)

# %%
res.message

# %%
with np.printoptions(precision=4):
    nl = "\n"
    print(f"min(-log L):              {res.fun}")
    print(f"Fit result [I_Tm1, I_T]: {str(res.x.reshape(2,-1)).replace(nl, ',')}")
    print("------------------------------------")
    print(f"Orig. -log L:             {cost_2t(x0)}")
    print(f"Orig. vals [I_Tm1, I_T]: [{np.array(Itilde_Tm1_evol[step_idx])}, {np.array(Itilde_T_evol[step_idx])}]")

# %% [markdown]
# Not only does the fit not explode, ~but it brings $\tilde{I}_T$ and $\tilde{I}_{T-1}$ back to sensible values.~
#
# At this point we can be pretty confident in the cost function provided to the optimizer. The problem is likely elsewhere in the optimization procedure.

# %% [markdown]
# ---
# ## Plotting likelihood profiles
# What does the likelihood look like ? Is it convex ? A little bumpy ? A *lot* bumpy ?
#
# [TODO]

# %%
import scipy.interpolate

# %%
set_to_step(-1)

# %%
model.Itilde.unlock()

# %%
model.theano_reset()

# %% [markdown]
# from IPython import get_ipython
# if get_ipython():
#     # Workaround to hide the extra space left by removed tqdm widgets
#     # https://github.com/jupyter-widgets/ipywidgets/issues/1845#issuecomment-594985543
#     from IPython.core.display import HTML, display
#     def rm_out_padding(): display(HTML("<style>div.output_subarea { padding:unset;}</style>"))
#     rm_out_padding()

# %%
import copy
from warnings import warn
import numpy as np
import scipy.interpolate
from collections.abc import Callable

import json
import mackelab_toolbox.typing as mtbtyping

import bokeh.plotting
from tqdm.auto import tqdm

class InterpLogp:
    """
    Class for arbitrary n-D function, which may be expensive to evaluate.
    Points at which it is evaluated are stored in a list.
    Provides two principal helper methods:
    - `interpolate`: Cheap evaluation at a new point by interpolating
      existing points.
    - `meshgrid`: Return interpolation in a format compatible with
      NumPy's `meshgrid`.
    """
    def __init__(self, f, rng=None):
        self.xlist = []
        self.zlist = []
        self.f = f
        self.rng = rng or np.random.default_rng(181121617223716287007530534937777931929)
        self.cleanup_callback = None
        self.liveplot_data = {}

    def save(self, file):
        """
        Output data to a file as a ``.npz`` NumPy archive.
        Not saved:
            - Evaluation function `f`
            - `cleanup_callback`.
        :param file: Passed on to `np.savez`.
        """
        rng_str = json.dumps(mtbtyping.RNGenerator.json_encoder(self.rng))
        np.savez(
            file,
            xlist=self.xlist,
            zlist=self.zlist,
            rng=rng_str
        )

    def load(self, file, replace=True):
        """
        Load previously saved InterpFunc data into this object.
        The evaluation function is not reloaded

        :param file: Passed on to `np.load`.
        :param replace:
            True: Replace any current data with loaded values.
                  RNG state is loaded from file.
            False: Append loaded values to the current data.
                  The RNG state is unchanged
                  (i.e. the one saved to file is ignored)
        """
        data = np.load(file)
        if replace:
            self.xlist = list(data['xlist'])
            self.zlist = list(data['zlist'])
            self.rng = mtbtyping.RNGenerator.validate(json.loads(data['rng'][()]))
        else:
            self.xlist.extend(data['xlist'])
            self.zlist.extend(data['zlist'])

    def eval(self, x):
        """(Possibly) expensive evaluation by calling the true function."""
        x = np.array(x)
        for x_, z in zip(self.xlist, self.zlist):
            if np.all(x == x_):
                return z
        self.xlist.append(x)
        self.zlist.append(self.f(x))
        return self.zlist[-1]
    def densify(self, n, sampler, *args,
                leave_progbar=True, position_progbar=None, progbar=None, **kwargs):
        """
        Extend the set of evaluated points by sampling `n` new points with
        `sampler(**kwargs)` and evaluating the function at those points.
        :param n: Number of new points to evaluate
        :param sampler: Callable | str
            If a str, must match one of the sampling methods of `numpy.random.Generator`.
        :param leave_progbar: Set to False to remove the progress bar once function completes.
        :param position_progbar: Passed as argument to create the tqdm progress bar
        :param progbar: Alternative to specifying progress bar parameters:
            Pass an existing tqdm progress bar to use.
            When this parameters is provided, `leave_progbar` and `position_progbar` are ignored.
        :param *args: Additional arguments are passed to the sampler
        :param **kwargs: Additional keyword arguments are passed to the sampler

        :returns: xlist: List of drawn arguments
        :returns: zlist: List of evaluations
        """
        if isinstance(sampler, str):
            sampler = getattr(self.rng, sampler)
        xlist = [sampler(**kwargs) for i in range(n)]
        if progbar:
            progbar.disable = False  # Disable is set to True when bar completes
            progbar.total = len(xlist)
            progbar.n = 0
            progbar.iterable = xlist
            progbar.refresh()
            def make_it():
                # HACK: We shouldn't need to call .refresh() – defeats the point of miniters
                for x in xlist:
                    progbar.update()
                    progbar.refresh()
                    yield x
            it = make_it()
        else:
            it = tqdm(xlist,
                      leave=leave_progbar, position=position_progbar)
        zlist = [self.f(x) for x in it]
        self.xlist.extend(xlist)
        self.zlist.extend(zlist)
        return xlist, zlist

    def argmax(self):
        """Return the argument vector which yielded the maximum function value"""
        return self.xlist[np.argmax(self.zlist)]
    def argmin(self):
        """Return the argument vector which yielded the minimum function value"""
        return self.xlist[np.argmin(self.zlist)]
    def interpolate(self, x):
        """(Hopefully) cheap evaluation by using interpolation."""
        return scipy.interpolate.griddata(self.xlist, self.zlist, xi=x)
    def meshgrid(self, *xi):
        """
        Same arguments as you would pass to meshgrid.
        :returns: N-D array, with shape give by outer product of xi
        """
        D = len(xi)
        xi_expanded = [None]*D
        for i, x in enumerate(xi):
            assert x.ndim == 1
            xi_expanded[i] = x[(None,)*(D-i-1) + (slice(None),) + (None,)*i]
        return scipy.interpolate.griddata(self.xlist, self.zlist, xi=tuple(xi_expanded))
    def automeshgrid(self, n):
        """
        Similar to `meshgrid`, but auto-determines the meshgrid along each dimension.
        :param n: int | tuple
            If int: Grid will have this many points along each axis
            If tuple: Must have length D; specifies number of points for each dimension.

        :returns size-2 tuple:
            xi: [list, size D]: array of stops for each axis
            z_meshgrid [ndarray]: meshgrid for z

            To convert stops to the appropriate meshgrid, can simply call `np.meshgrid(xi)`.
        """
        if len(self.xlist) == 0:
            warn("List of reference points is empty.")
            return
        x_arr = np.array(self.xlist)
        z_arr = np.array(self.zlist)
        if isinstance(n, int):
            n = [n]*x_arr.shape[1]
        xi_arrs = [np.linspace(min(xi_arr), max(xi_arr), ni)
                   for xi_arr, ni in zip(x_arr.T, n)]

        z_meshgrid = self.meshgrid(*xi_arrs)

        return np.array(xi_arrs), z_meshgrid

    ## Display / inspection ##

    def threshold(self, zmin=np.inf, zmax=np.inf):
        interp = copy.copy(self)
        zarr = np.array(interp.zlist)
        xarr = np.array(interp.xlist)
        mask = np.logical_and(zmin <= zarr, zarr <= zmax)
        interp.xlist = list(xarr[mask])
        interp.zlist = list(zarr[mask])
        return interp

    def update_liveplot_hist(self, fig, counts, edges):
        data={'top': counts,
              'left': edges[:-1],
              'right': edges[1:]}
        if fig.id not in self.liveplot_data:
            self.liveplot_data[fig.id] = bokeh.models.ColumnDataSource(
                data=data
            )
            # https://docs.bokeh.org/en/latest/docs/gallery/histogram.html
            fig.quad(top='top', bottom=0, left='left', right='right',
                     fill_color="navy", line_color="white", alpha=0.5,
                     source=self.liveplot_data[fig.id])
        else:
            self.liveplot_data[fig.id].data = data

    ## Log p methods ##
    def auto_densify(self, sampler, loc_initial, scale_initial, scale_decay=3,
                     n_per_iter=5, liveplot=True,
                     discard_fraction=0.5, draw_from_HPR=True,
                     required_points_in_HPR=30, HPR_boundary=0.01):
        """
        Repeatedly call `densify` until the estimated High Probability Region (HPR)
        contains sufficient number of points.
        On each iteration, a certain number of points are discarded, and from the
        remaining set, a reference point is drawn uniformly. The new points are
        drawn from a distribution centered around this reference.

        :param sampler: Callable | str
            If a str, must match one of the sampling methods of `numpy.random.Generator`.
            The sampler must take two arguments, `loc` and `scale`.
        :param loc_initial: ndarray
            The reference point to use for the first iteration.
        :param scale_initial: float | ndarray
            The initial value of the sampling distribution's 'scale' parameter.
        :param scale_decay: float | ndarray
            At every iteration, the scale is decreased by (1 - e(-scale_decay))
        :param n_per_iter: Number of points to evaluate at each iteration
        :param liveplot: bool
            True: Display a live plot of the distribution of points with the
            highest log p. The number of points is the same as `required_points_in_HPR`,
            such that the termination condition corresponds to these points
            all being close enough together.
        :discard_fraction: Number of points to discard at each iteration
            Ignored when `draw_from_HPR` is True.
        :draw_from_HPR: Restrict the selection of new points to the current
            HPR estimate. This can dramatically speed up exploration, at the
            cost of a higher likelihood of missed alternative modes.
        :required_points_in_HPR: Terminate when there are this many points
            in the estimated HPR
        :HPR_boundary: Points are considered part of the HPR when the
            ratio of their probability to the MLE is greater or equal to
            this value.
        """
        assert scale_decay > 0
        converged = False
        HPR_log_boundary = np.log(HPR_boundary)
        xarr = np.array(self.xlist)  # Make copies because we will discard low p values
        zarr = np.array(self.zlist)  # Also make arrays to allow fancy indexing
        loc = loc_initial
        scale = scale_initial
        HPR_progbar = tqdm(desc="HPR size", total=required_points_in_HPR, position=0)     # Track number of points in the HPR
        iter_progbar = tqdm(desc="Iterations", position=1)  # Track number of iterations
        samples_progbar = tqdm(desc="Evaluations (cur. iter.)", total=n_per_iter, position=2)
        if liveplot:
            fig = bokeh.plotting.figure(plot_width=300, plot_height=200)
            fig.y_range.start = 0
            fig.xaxis.axis_label = f"log p (highest {required_points_in_HPR} values)"
            fig.grid.grid_line_color="white"
            plot_zarr = zarr[-30:]
            if len(plot_zarr) < 0:
                plot_zarr = np.array([0])
            counts, edges = np.histogram(zarr[-required_points_in_HPR:],
                                         bins=required_points_in_HPR)
            self.update_liveplot_hist(fig, counts, edges)
            HPR_start = bokeh.models.annotations.Span(
                location=(zarr.max() - HPR_log_boundary) if zarr.size else 0,
                dimension='height',
                line_color='#888888', line_dash='dashed', line_width=1)
            fig.add_layout(HPR_start)
            liveplot_handle = bokeh.io.show(fig, notebook_handle=True)
        try:
            while not converged:
                # Move iterations progress bar forward by 1
                iter_progbar.update()
                # Compute new points
                new_xlist, new_zlist = self.densify(
                    n_per_iter, sampler, loc=loc, scale=scale,
                    progbar=samples_progbar)
                if len(xarr):
                    assert len(zarr)
                    xarr = np.concatenate((xarr, new_xlist))
                    zarr = np.concatenate((zarr, new_zlist))
                else:
                    xarr = np.array(new_xlist)
                    zarr = np.array(new_zlist)
                # Sort the new points based on their log p; discard lowest n from sampling
                n_discard = int(discard_fraction * len(zarr))  # Ensures at least one point is kept
                σ = np.argsort(zarr)[n_discard:]
                xarr = xarr[σ]
                zarr = zarr[σ]
                # Determine points in the estimated HPR
                HPR = [xarr[-1]]
                zmax = zarr[-1]
                for x, z in zip(xarr[-2::-1], zarr[-2::-1]):
                    if z - zmax < HPR_log_boundary:
                        break
                    HPR.append(x)
                # Refresh HPR progress bar
                HPR_progbar.n = len(HPR)
                HPR_progbar.refresh()
                # Update the plotted distribution of z-values
                if liveplot:
                    counts, edges = np.histogram(zarr[-required_points_in_HPR:],
                                                 bins=required_points_in_HPR)
                    self.update_liveplot_hist(fig, counts, edges)
                    HPR_start.location = zmax + HPR_log_boundary
                    bokeh.io.push_notebook(handle=liveplot_handle)
                # Decide if we have enough points
                if len(HPR) >= required_points_in_HPR:
                    converged = True
                    break
                # We don't have enough points => choose a new reference point
                if draw_from_HPR:
                    loc = self.rng.choice(HPR)
                else:
                    loc = self.rng.choice(xarr)
                scale *= (1 - np.exp(-scale_decay))
        except KeyboardInterrupt:
            if self.cleanup_callback:
                self.cleanup_callback()
        finally:
            iter_progbar.close()

    def get_HPR(self, p_boundary):
        """
        :p_boundary: Points are considered part of the HPR when the
            ratio of their probability to the MLE is greater or equal to
            this value.
        """
        log_boundary = np.log(p_boundary)
        xlist = self.xlist
        zlist = self.zlist
        σ = np.argsort(zlist)
        HPR = [self.xlist[σ[-1]]]
        zmax = zlist[σ[-1]]
        for i in σ[-2::-1]:
            if zlist[i] - zmax < log_boundary:
                break
            HPR.append(xlist[i])
        return HPR


# %%
logp_interp = InterpLogp(lambda Itilde: -cost_1t(Itilde))

# %%
logp_interp.load("IT_at_n-5000.npz")

# %%
model.theano_reset()

# %%
μ = model.Itilde.data.mean(axis=0)
σ = model.Itilde.data.std(axis=0)

# %%
logp_interp.auto_densify(
    sampler='normal', loc_initial=μ, scale_initial=σ,
    liveplot=True
)

# %%
from matplotlib.colors import ListedColormap

# %%
fig, ax = plt.subplots()
logp_thresh = logp_interp.threshold(-np.inf)
xi_arrs, logp_meshgrid = logp_thresh.automeshgrid(50)
cf = ax.contourf(*xi_arrs, logp_meshgrid,
                 #levels=(np.nanmax(logp_meshgrid)-relative_levels)[::-1]
                 #cmap=ListedColormap(sns.color_palette('rocket'))
                 );
ax.set_xlabel("$\\tilde{I}_{T}^1$")
ax.set_ylabel("$\\tilde{I}_{T}^2$")

ax.scatter(*np.array(logp_thresh.xlist).T,
           c=logp_thresh.zlist,
           cmap=ListedColormap(sns.color_palette('rocket')))

cbar = fig.colorbar(cf, ax=ax);
cbar.set_label("$\log L$");

# %% [markdown]
# ---

# %%
import numpy as np

# Define the dimensionality of our problem.
ndim = 2

# Define our 3-D correlated multivariate normal likelihood.
C = np.identity(ndim)  # set covariance to identity matrix
C[C==0] = 0.95  # set off-diagonal terms
Cinv = np.linalg.inv(C)  # define the inverse (i.e. the precision matrix)
lnorm = -0.5 * (np.log(2 * np.pi) * ndim +
                np.log(np.linalg.det(C)))  # ln(normalization)

def loglike(x):
    """The log-likelihood function."""

    return -cost_1t(x)

# Define our uniform prior.
def ptform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""

    return 100. * (2. * u - 1.)


# %%
import dynesty

# "Static" nested sampling.
sampler = dynesty.NestedSampler(loglike, ptform, ndim)
sampler.run_nested()
sresults = sampler.results

# %%

# %%
