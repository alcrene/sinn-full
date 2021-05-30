# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # Sampling algorithms
#
# This module defines and describes the functions used to sample the data.
#
# :::{caution}
# The functionality in this module may be changed in the future to more closely align with the API of [_torch.utils.data_](https://pytorch.org/docs/stable/data.html).
# :::

# %%
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')

# %%
import abc
import numpy as np
import itertools
import inspect
from numbers import Integral
from pydantic import BaseModel
import mackelab_toolbox as mtb
import mackelab_toolbox.iotools
from mackelab_toolbox.units import get_magnitude
from sinn.histories import TimeAxis
from sinn.utils.pydantic import initializer

from sinnfull import ureg
from sinnfull.rng import get_np_rng

default_rng = np.random.default_rng()

# %% [markdown]
# ## Sampling data segments

# %% [markdown]
# ### Baseline data
#
# For training the baseline model, we want segments that are far enough from the seizure onset. Ideally we use the same amount of data from each trial, to avoid biasing the fit; and since the seizure in *NA_d2_sz1* starts at 86s, segments of 50s per trial seem like a good choice. We align each segment with the start of the recording: since that is uncorrelated with the start of the seizure, it's as good a choice as any.
#
# These training segments are indicated by the shaded grey portion in the figure below.

# %%
from sinnfull.data import DataAccessor   # These imports only used for type hints
from mackelab_toolbox.typing import PintValue, RNGenerator
from typing import Union, Tuple, List, Dict, Callable

# %%
class SegmentSampler(BaseModel, abc.ABC):
    """
    The base class for segment samplers. The actual sampling function is
    implemented by subclasses, in their `draw` method.

    Implementation of `draw` in subclasses
    --------------------------------------
    Subclasses may define any number of additional attributes as annotations,
    and access them through `self`.

    The `draw` implementation must take a single argument `rng` and use it
    for any random function. The `data` and `trial_filter` attributes, as well
    as any additional ones, are accessible through `self`.

    .. Important:: To ensure reproducible draws, the `draw` function should not
       be called directly. One should rather use this object as an iterator::

           segment_sampler = SegmentSamplerSubclass(...)
           for segment in segment_sampler:
               ...
               if condition:
                   break

       Note that the iterator returned by the sampler is infinite, which is
       why the loop above includes an break condition.

    `draw` must return a length 3 tuple, composed of a unique trial key,
    the time slice corresponding to the segment, and the segment data.

    Parameters
    ----------
    data:
        A DataAccessor object
    trial_filter:
        Can be used to select which trials in `data` may provide data segments.
        By default, any trial in `data` is eligible; trials are sampled with
        equal probability, regardless of their length.
        This should be in a format understood by :meth:`sel`, and is passed as follows:
        ``data.trials.trial.sel(trial_filter)``.
    rng_key:
        The RNG used to sample trials.
    """
    data        : DataAccessor
    trial_filter: dict=None
    rng_key     : Union[Tuple[int], int]
    rng         : RNGenerator=None

    class Config:
        arbitrary_types_allowed = True

    @initializer('trial_filter')
    def default_trial_filter(cls, trial_filter):
        # intializers are only called if value is equal to None
        return {}

    @initializer('rng')
    def create_rng(cls, rng, rng_key):
        return get_np_rng(rng_key, exists_ok=True)

    # Register subclasses so they can be deserialized
    def __init_subclass__(cls):
        if not inspect.isabstract(cls):
            mtb.iotools.register_datatype(cls)

    # __new__ and __init__ serve only to allow deserializing more specialized
    # types, as determined by the output
    def __new__(cls, SamplerClass=None, **kwargs):
        """
        Allow instantiating a more specialized Sampler from the base class.
        If `SamplerClass` is passed (usually a string from the serialized model),
        the corresponding model type is loaded from those registered with
        mtb.iotools. Class construction is then diverted to that subclass.
        """
        if SamplerClass is not None:
            if isinstance(SamplerClass, str):
                try:
                    SamplerClass = mtb.iotools._load_types[SamplerClass]
                except KeyError as e:
                    raise ValueError(f"Unrecognized Sampler type '{SamplerClass}'."
                                     ) from e
            assert isinstance(SamplerClass, type) and issubclass(SamplerClass, cls)
            # IMPORTANT: __init__ will still be called with original sig,
            # so we still need to subclass it to consume the SamplerClass argument
            return SamplerClass.__new__(SamplerClass, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(cls, SamplerClass=None, **kwargs):
        super().__init__(**kwargs)

    # Declare __call__ an abstract method, to force subclasses to define it
    @abc.abstractmethod
    def draw(self, rng):
        pass

    def __iter__(self):
        return self
    def __next__(self):
        return self.draw(self.rng)

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        d['SamplerClass'] = mtb.iotools.find_registered_typename(type(self))
        return d

# %%
class FixedSegmentSampler(SegmentSampler):
    """
    Select one of the trials randomly (uniformly), and return the data
    segment sliced between `t0` and `t0 + T`.

    The window `[t0, t0+T]` is fixed at instance creation, hence the name.

    Parameters
    ----------
    data:
    trial_filter:
    rng_key:
        See SegmentSampler
    t0:
        Start point of segments, in seconds.
    T:
        Length of segments, in seconds.

    Returns
    -------
    trial_key: tuple
        A unique hashable key identifying the trial.
    segment_slice: slice
        The time-array slice used to obtain the segment from the trial.
        Applying this to `data` should be idempotent.
    data: xarray
        The segment data.
    """
    t0          : PintValue
    T           : PintValue

    def draw(self, rng):
        data=self.data; trial_filter=self.trial_filter
        t0=self.t0; T=self.T

        trial = rng.choice(data.trials.trial.sel(trial_filter))
        segment = data.load(trial)
        time_unit = segment.time.unit
        if isinstance(time_unit, str):
            time_unit = ureg(time_unit)
        t0 = t0.astype(segment.time.dtype)  # Especially important if t0
        T  =  T.astype(segment.time.dtype)  # and T are specified as ints
        t0 = get_magnitude(t0, in_units_of=time_unit)
        T  = get_magnitude(T,  in_units_of=time_unit)
        tslice = slice(t0, t0+T)
        return trial.key, tslice, segment.sel(time=tslice)


# %% [markdown]
# ### Random data

# %%
import xarray as xr
from pydantic import validator, root_validator
class RandomSegmentSampler(SegmentSampler):
    """
    Return completely random data with the same shape as the given time array.
    The data are draw by the specified distribution.

    Parameters
    ----------
    rng_key:

    var_dists: Dictionary of var name: var dist.
        Variable names must match those in `var_shapes`.
        Variable distributions may be specified either as names or
        argument-less functions which return random numbers.

    var_shapes: Dictionary of var name: var shape
        Variable names must match those in `var_dists`

    .. warning:: In the current implementation, if variable distributions are
       specified as callables, they will not use the RNG associated to `rng_key`
       and the samples will not be reproducible.
    """
    data : int=0   # Just here to hide the base class parameter
    var_dists  : Dict[str,Union[str, Callable]]
    var_shapes : Dict[str,Tuple[int,...]]
    time : TimeAxis


    @initializer('var_dists', always=True)
    def get_dists(cls, var_dists, rng):
        for k, distname in var_dists.items():
            if isinstance(distname, str):
                var_dists[k] = getattr(rng, distname)
        return var_dists

    @root_validator
    def names_consistent(cls, values):
        dists = values.get('var_dists'); shapes = values.get('var_shapes')
        if set(dists) != set(shapes):
            raise ValueError("Distribution and shape keys don't match.\n"
                             f"Distribution keys: {sorted(set(dists))}\n"
                             f"Shape keys: {sorted(set(shapes))}")
        return values

    def draw(self, rng):
        dists=self.var_dists; shapes=self.var_shapes; time=self.time

        time_array = time.stops_array
        # Draw a random key so each draw is different
        key = (rng.integers(10000),)
        # Time slice is always the same
        tslice = np.s_[time.t0:time.tn+time.dt]
        # Draw new random data
        data_vars = {k: xr.DataArray(
                        data=dists[k](size=(len(time),)+tuple(shapes[k])),
                        coords={'time':time_array},
                        dims=['time'] + [f'{k}_{i}' for i in range(len(shapes[k]))],
                        name=k
                        )
                     for k in dists}

        return key, tslice, xr.Dataset(data_vars)

# %%
if __name__ == "__main__":
    from sinn.histories import TimeAxis

    sampler = RandomSegmentSampler(rng_key=(3,),
                                   var_dists={'x': 'normal',
                                              'y': 'poisson'},
                                   var_shapes={'x': (2,), 'y': (3,)},
                                   time=TimeAxis(min=0, max=1, step=2**-5))

    for s, _ in zip(sampler, range(4)):
        print(s)

# %% [markdown]
# ## Sampling batches
#
# We want to select batches that provide good coverage of the data, without introducing artificial boundaries (such as would be the case if the data were just cut at $K_b$ intervals). The procedure used in (René et al. 2020) was to add a random spacing between each cut. This seems to work, but has a few drawbacks:
#
# - The sampling of time points is not uniform, especially at the beginning.
# - The position of a time point within a batch is even less uniform (time points are biased towards the beginning or end of a batch).
# - To sample the data more sparsely, the mean spacing between batches needs to be increased. This makes the sampling even less uniform.
# - The batch selection is tightly integrated in the inference loop, meaning more work when porting to a new problem.
#   In particular, the implementation was specific to a forward pass on the data.
# - One does not control exactly the number of samples obtained from one pass.
#
# The algorithm we propose here is mathematically simpler and has perfect uniformity away from edges. Also, since batches can be sampled in any order, it can be used for both a forward and backward pass. The idea is to draw the start times $b_i$ from a distribution $p_{{}_b}$. After every draw, this distribution is reduced around $b_i$ by a suppression factor $γ_{{}_b}$: $p_{{}_b} \leftarrow γ_{{}_b} p_{{}_b}$. In our implementation, $γ_{{}_b}$ is an inverted Gaussian.
#
# Since this algorithm is translation invariant, time points are sampled uniformly, independent of the suppression vector. Moreover, their position within a batch is also uniform.

# %% [markdown]
# ### Algorithm: `sample_batch_starts`$(K, K_b, N_b)$
#
# | Symbol | Identifier | Meaning |
# |--------|------------|---------|
# |$K$     | `K`        | Number of time bins in the data segment |
# |$K_b$   | `Kb`       | Batch size in time bins|
# |$N_b$   | `Nb`       | Number of batches to select
#
# 1. Compute the suppression array $γ_{{}_b}$:
#    \begin{align*}
#    γ_{{}_b k} &\leftarrow \mathop{PDF}_{\mathcal{N}(K_b, K_b^2/4)}(k)\,,\quad k=1,\dotsc,2K_b \,. \\
#    γ_{{}_b k} &\leftarrow (γ_{{}_b K_b} - γ_{{}_b k}) / γ_{{}_b K_b}
#    \end{align*}
#    (The second operation flips the Gaussian, so that it is zero at the centre $K_b$, and tapers off to one at the edges $γ_{{}_b 0}$ and $γ_{{}_b 2K_b}$
# 2. Initialize the draw probabilities $p_{{}_b,k} = 1$, $k = 1,\dotsc,K-K_b$.
# 3. For $i$ in $1,\dotsc,N_b$:
#     1. Normalize the draw probabilities: $p_{{}_b k} = p_{{}_b k} / \sum_k p_{{}_bk}$.
#     2. Draw  batch start time $b_i$ from the distribution with probabilities $p_{{}_b}$.
#     3. Suppress the draw probability around $b_i$:
#        $p_{{}_b\,b_i-K_b:b_i+K_b} \leftarrow γ_{{}_b} p_{b_i-K_b:b_i+K_b}$.

# %% [markdown]
# #### Numerical consideration: avoiding underflow
# If $K$ or $N_b$ are large enough, it can happen that computing $p_{{}_b k} = p_{{}_b k} / \sum_k p_{{}_bk}$ leads to some $p_{{}_bk}$ so small as to be numerically rounded to exactly zero. This is undesirable, since the value of that $p_{{}_bk}$ is then stuck at 0, and cannot rise again when batches are drawn from other positions. Eventually, *all* the $p_{{}_bk}$ become zero, and the algorithm breaks.
#
#
# Our solution to this problem is to add to the probabilities a small amount $ε$ after their normalization, between the steps 3A and 3B above:
#
# 3. &nbsp;
#     2. Ensure all probabilities are strictly positive: \
#        $p_{{}_bk} = p_{{}_bk} + ε$ \
#        $p_{{}_bk} = p_{{}_bk} / (1 + Kε)$
#
# We require that $Kε < 1$, which ensures that this second normalization is numerically stable and all probibilities are strictly positive. This step effectively introduces a probability floor $p_{\mathrm{min}}$, which is given by
#     $$p_{\mathrm{min}} = \frac{ε}{1 + Kε} \,.$$
# The function `sample_batch_starts` accepts a `p_min` arguments, from which we calculate the appropriate $ε$.

# %%
import numpy as np
import scipy.stats
def sample_batch_starts(K: int, Kb: int, Nb: int,
                        return_p: bool=False, init_p :np.ndarray=None,
                        rng: np.random.Generator=default_rng,
                        p_min: float=1e-6):
    """
    :param K:  Number of time bins.
    :param Kb: Size of a batch.
    :param Nb: Number of batches to select.
    :param return_weights: If true, also return the final weight vector for the time points
    :param init_p: If provided, initialize draw probabilities with this vector instead of ones.
        The array is updated in place, so make a copy before passing if needed.
    :param rng: The RNG to use to sample batches. If not provided, a default one is used.
    :param p_min: The minimum probability for each time point. For numerical
        reasons, it is essential to set this to a non-zero floor. This number
        also sets an upper limit on the number of time points `K`, which must
        be less than ``1/p_min``.
    """
    if Kb > K:
        raise ValueError(f"Batch size (Kb={Kb}) is larger than the number of "
                         f"time bins (K={K}).")
    assert isinstance(K, Integral)
    # Compute the increment used to enforce the probability floor
    ε = p_min / (1 - K*p_min)
    if ε <= 0:
        raise ValueError(f"Cannot sample batches with {K} time bins (K) and a "
                         f"probability floor of {p_min} (p_min). Either lower "
                         "the floor, or decrease the number of time bins.")
    barr = np.empty(Nb, dtype=np.min_scalar_type(K))
    # Initialize batch weights to uniform
    if init_p is not None:
        assert len(init_p) == K-Kb+1
        pb = init_p
    else:
        pb = np.ones(K-Kb+1)
    # Create the discount array
    γ = scipy.stats.norm(loc=Kb, scale=Kb/2).pdf(np.arange(2*Kb))
    γ = (γ.max()-γ) / γ.max()
    # Draw Nb batches
    for i in range(Nb):
        pb /= pb.sum()
        # Enforce probability floor. This must be done in 2 steps, to avoid
        # numerical rounding making some values 0
        pb += ε          # No numerical issue => stricly positive
        pb /= pb.sum()   # ~(1+K*ε) ~ 1 => no numerical issue => stays strictly positive
                         # Although we know the analytic form of pb.sum(), summing
                         # numerically corrects for any numerical errors
        # Draw a new batch start point
        b = rng.choice(K-Kb+1, p=pb)
        # Depress the draw probability around that point
        trunc_start = max(0, Kb-b)            # Truncation of the draw probability
        trunc_end = max(K-Kb+1, b+Kb)-K+Kb-1  # only occurs for draws at the edges
        pb[b-Kb+trunc_start:b+Kb] *= γ[trunc_start:len(γ)-trunc_end]
        barr[i] = b
    if return_p:
        return np.sort(barr), pb
    else:
        return np.sort(barr)


# %% [markdown]
# Below is an illustration of the batch sampler. We select 10 batches of length 10 out of the interval [0,100); compared to sampling batches uniformly, we see that `sample_batch_starts` produces batches with better coverage of the data interval. After sampling a batch at position $b$, the selection probabilities are updated by multiplying those in the vicinity of $b$ by $γ$, shown on the right. The orange line shows the selection probability function after all samples have been drawn.

# %%
if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Show an example draw
    K = 100
    Kb = 10
    Nb = 10
    Δ = K / 10
    ticks = np.arange(K//10+1)*Δ
    with sns.axes_style('ticks'):
        fig, (ax, ax_γ) = plt.subplots(1,2,figsize=(8,1.5), gridspec_kw={'width_ratios': [3, 1]})
        for ax_ in (ax, ax_γ):
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
        for k in ticks:
            ax.axvline(k, 0, 0.7)
        ax.set_xticks(ticks);
        ax.set_yticks([]);
        ax.set_xlim(ticks[0], ticks[-1]);
        batch_starts, draw_weights = sample_batch_starts(K, Kb, Nb, return_p=True)
        for b in batch_starts:
            ax.axvspan(b, b+Kb, 0, 0.5, alpha=0.3, color=sns.color_palette()[3])
        ax.plot(draw_weights, color=sns.color_palette()[1], label="$p(b_i=k)$");
        ax.legend(loc='upper left', bbox_to_anchor=(0,1.1));
        ax.set_title("Batch distribution (sample_batch_starts))")
        ax.set_xlabel("$k$")

        γ = scipy.stats.norm(loc=Kb, scale=Kb/2).pdf(np.arange(2*Kb))
        γ = (γ.max()-γ) / γ.max()
        ax_γ.plot(γ)
        ax_γ.set_yticks([])
        ax_γ.set_xticks([0, Kb, 2*Kb-1])
        ax_γ.set_xticklabels(["1", "$K_b$", "$2 K_b$"])
        ax_γ.set_title("$γ_k$")
        ax_γ.set_xlabel("$k$")
        plt.show()

        # Uniform sampling
        fig, (ax, ax_γ) = plt.subplots(1,2,figsize=(8,1.5), gridspec_kw={'width_ratios': [3, 1]})
        ax_γ.set_axis_off()
        for ax_ in (ax, ax_γ):
            ax_.spines['top'].set_visible(False)
            ax_.spines['right'].set_visible(False)
            ax_.spines['left'].set_visible(False)
        for k in ticks:
            ax.axvline(k, 0, 0.7)
        ax.set_xticks(ticks);
        ax.set_yticks([]);
        ax.set_xlim(ticks[0], ticks[-1]);
        batch_starts = default_rng.choice(K, Nb)
        for b in batch_starts:
            ax.axvspan(b, b+Kb, 0, 0.5, alpha=0.3, color=sns.color_palette()[3])

        ax.set_title("Batch distribution (uniform)")
        ax.set_xlabel("$k$");
        plt.show()


# %% [markdown]
# **Test**: All possible start values are eventually drawn

# %%
if __name__ == "__main__":
    K = 100
    Kb = 10
    N = 200
    starts = set(sample_batch_starts(K, Kb, N))
    assert starts == set(range(K - Kb+1))
