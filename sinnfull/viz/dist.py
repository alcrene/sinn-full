# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (sinnfull)
#     language: python
#     name: sinnfull
# ---

# %% [markdown]
# # Visualizing distributions
#
# :::{note}  
# The functions in this module do not depend on other sinnfull libraries, so they can be easily used in other projects.  
# :::

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')

# %%
import copy
from time import time
from warnings import warn
import numpy as np
import scipy.interpolate
from collections.abc import Callable

import json
import mackelab_toolbox.typing as mtbtyping

import bokeh
import bokeh.io
import bokeh.plotting
from tqdm.auto import tqdm

# %% tags=["remove-cell"]
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import seaborn as sns
    import sinnfull.viewing  # Load plotting defaults
    bokeh.io.output_notebook()


# %% [markdown]
# ## Interpolated scalar function
#
# The class `InterpScalarFunc` is intended for *visualization* and *exploration* of a possibly high-dimensional, computationally expensive function; the model case is an unnormalized log likelihood. The basic interface is composed of the following methods:
# - *visualization*:  
#   When provided with a limited set of evaluation points, these methods provide an expedient way of visualizing the function. They are most useful when the points are relatively sparse.
#   - `meshgrid`: Provide the same arguments to `numpy.meshgrid` and `InterpLogp.meshgrid`  
#     The former returns the domain, the latter the image of the function.
#   - `automeshgrid`: A convenience function. Provide the discretization level for each dimension, and the method returns meshgrids for both the domain and image.
# - *exploration*:  
#    The combination of `densify` and `threshold` can be used to “zoom in” to the distribution, without wasting computational resources on the region of interest. In contrast to statistically sound sampling like MCMC or slice sampling, this can may be done in a matter of seconds, allowing for use in an interactive session.
#   - `densify`: Increase the resolution of the interpolated function around a point of interest.
#   - `threshold`: Return a copy of `InterpScalarFunc` where only points which evaluate above the given threshold are kept.
#
# For (unnormalized) lop probabilities, the specialized class `InterpLogp` provides two additional methods:
#
# - `autodensify`: Draw samples from the distribution until a high probability region is sampled to some degree of accuracy.
# - `get_HPV`: Return the samples within the high probability region, defined by having their relative probability density to the mode greater than a certain threshold.
#
# :::{admonition} Caution: **`autodensify` is not a distribution sampler.**
# :class: caution
#
# The selected sampling points have *no* useful statistics: they are sampled from an arbitrary cheap distribution. They are *only* valid for interpolating the function of interest. Moreover, there is *no guarantee* that all modes of a distribution would be found, even asymtotically. For any other use (computing expections, model evidence…), use a proper sampling algorithm like MCMC or slice sampling. In such a case, the visualization methods may still be used to view the resulting distribution.  
# :::
#
# :::{note}
# Also note that the definition of the HPV by relative probability density means the probability volume it contains is undetermined, and in some pathological cases may be very small. This choice is made in line with the goal of expediency over correctness.  
# :::
#
# **Why not just use a proper sampler ?**
# - As noted above, even with statistically meaningful samples, one still needs a way to display these as a distribution. Binning the samples into *n-D* histograms is the typical way to do this. Interpolating the *log p* is an alternative approach, which is especially interesting if the samples are sparse (e.g. far from the HPV).
# - By their nature, samplers must always draw samples proportionally to the probability density. If the goal is to *visualize* the density, this is inefficient: even when the HPV is well sampled at the desired “zoom level”, most points will continue to be drawn from the HPV rather than the undersampled low-probability regions.

# %%
class InterpScalarFunc:
    # TODO: Move to ndfunc module
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
        """
        Parameters
        ----------
        f: Callable (ndarray) -> float
            The function to interpolate.
        rng: (Optional) Generator | int
            If not provided, a default rng is created with `numpy.random.default_rng`.
            If an int, used as a seed for `default_rng`.
        """
        self.xlist = []
        self.zlist = []
        self.f = f
        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        self.cleanup_callback = None
        self.liveplot_data = {}

    def __len__(self):
        assert len(self.xlist) == len(self.zlist)
        return len(self.xlist)

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
            progbar.miniters = 0
            progbar.avg_time = None
            progbar.last_print_n = 0
            progbar.last_print_t = progbar.start_t = time()
            progbar.refresh()
            def make_it():
                # HACK: We should let tqdm deal with updates – here we copied some
                # code from std_tqdm.__iter__
                n = progbar.n
                for x in xlist:
                    n += 1
                    Δn = n - progbar.last_print_n
                    if Δn >= progbar.miniters:
                        cur_t = time()
                        delta_t = time() - progbar.last_print_t
                        if delta_t >= progbar.mininterval:
                            #self.n = n
                            progbar.update(Δn)
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



# %%
class InterpLogp(InterpScalarFunc):

    ## Log p methods ##
    def auto_densify(self, loc_initial, scale_initial,
                     sampler='normal',
                     scale_decay=1, n_per_iter=5, liveplot=True,
                     discard_fraction=0.5, draw_from_HPV=True,
                     max_iterations=np.inf,
                     required_points_in_HPV=30, HPV_boundary=0.01):
        """
        Repeatedly call `densify` until the estimated High Probability Volume (HPV)
        contains sufficient number of points.
        On each iteration, a certain number of points are discarded, and from the
        remaining set, a reference point is drawn uniformly. The new points are
        drawn from a distribution centered around this reference.

        :param loc_initial: ndarray
            The reference point to use for the first iteration.
        :param scale_initial: float | ndarray
            The initial value of the sampling distribution's 'scale' parameter.
        :param sampler: Callable | str  (Optional)
            If a str, must match one of the sampling methods of `numpy.random.Generator`.
            The sampler must take two arguments, `loc` and `scale`.
            Default: 'normal'
        :param scale_decay: float | ndarray (default: 1)
            At every iteration, the scale is decreased by (1 - e(-1/scale_decay))
            A higher rate trades exploitation (looking in the neighbourhood of high
            value points) over exploration (seeing if there are other high value
            points elsewhere).
            Setting to zero prevents any decay, but makes it very difficult to
            resolve tighter peaks.
        :param n_per_iter: Number of points to evaluate at each iteration
        :param liveplot: bool
            True: Display a live plot of the distribution of points with the
            highest log p. The number of points is the same as `required_points_in_HPV`,
            such that the termination condition corresponds to these points
            all being close enough together.
        :discard_fraction: Number of points to discard at each iteration
            Ignored when `draw_from_HPV` is True.
        :draw_from_HPV: Restrict the selection of new points to the current
            HPV estimate. This can dramatically speed up exploration, at the
            cost of a higher likelihood of missed alternative modes.
        :max_iterations: Stop sampling after this many iterations.
            Default is to sample forever until convergence.
        :required_points_in_HPV: Terminate when there are this many points
            in the estimated HPV
            To deactivate this condition and ensure exactly `max_iterations`
            are performed, set to `numpy.inf`.
        :HPV_boundary: Points are considered part of the HPV when the
            ratio of their probability to the MLE is greater or equal to
            this value.
        """
        assert scale_decay >= 0
        converged = False
        HPV_log_boundary = np.log(HPV_boundary)
        # Ensure the discard fraction is not so high that we never keep enough points to converge.
        discard_fraction = min(discard_fraction, 1/(1 + required_points_in_HPV/n_per_iter))
        xarr = np.array(self.xlist)  # Make copies because we will discard low p values
        zarr = np.array(self.zlist)  # Also make arrays to allow fancy indexing
        loc = loc_initial
        scale = scale_initial
        HPV_progbar = tqdm(desc="HPV size", total=required_points_in_HPV, position=0)     # Track number of points in the HPV
        iter_progbar = tqdm(desc="Iterations", position=1)  # Track number of iterations
        samples_progbar = tqdm(desc="Evaluations (cur. iter.)", total=n_per_iter, position=2)
        if liveplot:
            fig = bokeh.plotting.figure(plot_width=300, plot_height=200)
            fig.y_range.start = 0
            fig.xaxis.axis_label = f"log p (highest {required_points_in_HPV} values)"
            fig.grid.grid_line_color="white"
            plot_zarr = zarr[-30:]
            if len(plot_zarr) < 0:
                plot_zarr = np.array([0])
            counts, edges = np.histogram(zarr[-required_points_in_HPV:],
                                         bins=required_points_in_HPV)
            self.update_liveplot_hist(fig, counts, edges)
            HPV_start = bokeh.models.annotations.Span(
                location=(zarr.max() - HPV_log_boundary) if zarr.size else 0,
                dimension='height',
                line_color='#888888', line_dash='dashed', line_width=1)
            fig.add_layout(HPV_start)
            liveplot_handle = bokeh.io.show(fig, notebook_handle=True)
        try:
            iteration_i = 0
            while not converged and iteration_i < max_iterations:
                iteration_i += 1
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
                # Determine points in the estimated HPV
                HPV = [xarr[-1]]
                zmax = zarr[-1]
                for x, z in zip(xarr[-2::-1], zarr[-2::-1]):
                    if z - zmax < HPV_log_boundary:
                        break
                    HPV.append(x)
                # Refresh HPV progress bar
                HPV_progbar.n = len(HPV)
                HPV_progbar.refresh()
                # Update the plotted distribution of z-values
                if liveplot:
                    counts, edges = np.histogram(zarr[-required_points_in_HPV:],
                                                 bins=required_points_in_HPV)
                    self.update_liveplot_hist(fig, counts, edges)
                    HPV_start.location = zmax + HPV_log_boundary
                    bokeh.io.push_notebook(handle=liveplot_handle)
                # Decide if we have enough points
                if len(HPV) >= required_points_in_HPV:
                    converged = True
                    break
                # We don't have enough points => choose a new reference point
                if draw_from_HPV:
                    loc = self.rng.choice(HPV)
                else:
                    loc = self.rng.choice(xarr)
                if scale_decay:
                    scale *= (1 - np.exp(-scale_decay))
        except KeyboardInterrupt:
            if self.cleanup_callback:
                self.cleanup_callback()
        finally:
            iter_progbar.close()

    def get_HPV(self, p_boundary):
        """
        :p_boundary: Points are considered part of the HPV when the
            ratio of their probability to the MLE is greater or equal to
            this value.
        """
        log_boundary = np.log(p_boundary)
        xlist = self.xlist
        zlist = self.zlist
        σ = np.argsort(zlist)
        HPV = [self.xlist[σ[-1]]]
        zmax = zlist[σ[-1]]
        for i in σ[-2::-1]:
            if zlist[i] - zmax < log_boundary:
                break
            HPV.append(xlist[i])
        return HPV


# %% [markdown]
# ### Example
#
# As an example of a computationally expensive distribution, we define the following:
#
# - Take $c, x \in \mathbb{R}^2$ and $f(x) = \begin{pmatrix}x_0^2 - x_1^2 \\ 2 x_0 x_1\end{pmatrix} + c \,.$
# - Define $\mathcal{D}_n(s)$, $n \in 1, 2, \dotsc$ to be the distribution with the p.d.f. proportional to :
#     $$p(c) \propto \exp\bigl[- \lVert \underbrace{f( \dotso f}_{\text{$n$ times}}(s)\dotso)\rVert_1 \bigr]$$

# %%
if __name__ == "__main__":

    c = 1
    n = 4

    domain_center=np.array([0.,0.])
    domain_radius=10.

    def logp(x):
        def f(x):
            return np.stack((x[...,0]**2 - x[...,1]**2, 2*x[...,0]*x[...,1]), axis=-1) + c
        for i in range(n):
            x = f(x)
        def norm(_x):
            return np.sum(abs(_x), axis=-1)
        return (4 - norm(x)) - np.sum((c-domain_center)**2)/(2*domain_radius)
        #bound = SmoothClip(4.)
        #flatten_at_zero = FlatLowerBound(0., 0.001)
        #return ((bound.high - bound(flatten_at_zero(norm(x))))
        #        - (np.sum((c-domain_center)**2)/(2*domain_radius)))

    class SmoothClip:
        """
        Smooth, symmetric clippin function using the hyperbolic tangent.
        Takes a single parameter, Δ, which sets both the expected domain
        and the value of the bounds.

        Parameters:
        Δ: Positive number.
           The smoother is approximately the identity on [-Δ, Δ].
           Function will be bounded by (-2Δ, 2Δ).
        """
        def __init__(self, Δ):
            self.Δ = Δ
        def __call__(self, x):
            Δ = self.Δ
            return (2*Δ) * np.tanh(x/(2*Δ))
        @property
        def low(self):
            return -2*self.Δ
        @property
        def high(self):
            return 2*self.Δ

    class FlatLowerBound:
        """
        Make a boundary flat (i.e. its derivative zero). Function is only valid on one side
        of the bound.

        **Important**: This does not enforce the bound. If you
        can't guarantee that x remains on the correct side of
        the bound, combine it with clipping:

        >>> smooth = SmoothLowerBound(0)
        >>> result = smooth(clip(x, 0, None))
        """

        def __init__(self, bound, β):
            """
            :param bound: Bound at which we want a flat derivative.
            :param β: Scale over which the input is distorted to achieve flat derivative.
            :param side: Either 'upper' or 'lower'.
            """
            self.bound = bound
            self.β = β
            self.a = -1  # lower bound

        def __call__(self, x):
            β = self.β
            a = self.a
            if (x == self.bound).all:
                return np.broadcast_to(self.bound + β, x.shape)
            else:
                return x + β*np.exp(a*x/β)


# %% [markdown]
# Create the sampler/interpolator

    # %%
    logp_interp = InterpLogp(logp)

# %% [markdown]
# Add samples around the HPV (high-probability volume). Re-run the cell to add more points.

    # %%
    logp_interp.auto_densify(sampler=np.random.normal,
                             n_per_iter=5,
                             scale_decay=0,
                             draw_from_HPV=False,
                             loc_initial=[0.,0.], scale_initial=1.)

# %% [markdown]
# Plot the distribution by interpolating between sampled points.

    # %%
    fig, ax = plt.subplots()
    # Create a new InterpLogp, with only points above a certain threshold
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
# `zlist` is the list of function evaluations (i.e. the list of _log L_ values).
# Display the 30 highest values.

    # %%
    sorted(logp_interp.zlist)[:-30:-1]

# %% [markdown]
# Almost all the points are concentrated in the HPV

    # %%
    plt.hist(logp_interp.zlist, bins=100);
    plt.xlabel("$\log L$");
    plt.ylabel("Counts");

    # %%
    len(logp_interp.zlist)

# %%
