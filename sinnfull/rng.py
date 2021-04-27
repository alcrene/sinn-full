# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %%
from __future__ import annotations

# %% [markdown]
# # Managing random numbers
#
# In order to perform reproducible inference experiments, we need reproducible random numbers. Moreover, we want these numbers to be high-quality: two numbers in the sequence should be as statistically independent as possible.
#
# The usual recommended way to do this is to seed one global random number generator (RNG) with a fixed value and use that for all random numbers. To see why this is not ideal, consider the typical situation where we have a stochastic model $\mathcal{M}_θ$ which depends on some noise input $ξ$, and a fitting algorithm $\mathcal{A}$ which requires an initial guess for the parameters $θ$. We want to test the algorithm for both different initial parameters $θ$ and different noise realizations $ξ$. The most convenient way to specify both, without introducing a bias (e.g. towards integer or repeated parameter values), is specify them as seeds, which an RNG can convert to the appropriate shape. However, changing the seed of $ξ$ should not change the values drawn for $θ$, and vice-versa. We _could_ simply define an RNGs for each seed, but these would have no guarantee of being independent – especially if low-entropy seeds are used, e.g. 0, 1, 2… What we want then is the following:
#
# - To convert low-entropy keys into high-quality RNG seeds;
# - To generator multiple independent and (very probably) non-overlapping RNGs from different seeds;
# - To make it simple to define different sequences of seeds for different purposes (e.g. initializing $θ$ vs generating $ξ$).
#
# Our approach is the following:
#
# - Rather than _integers_ as RNG _seeds_, we use _tuples_ as RNG _keys_.
# - RNG keys are of the form _(purpose, seed)_, where both _purpose_ and _seed_ are integers. For example, the _purpose_ integers for $θ$ and $ξ$ might respectively be 0 and 1. RNG keys for $θ$ would then have the form `(0, 0)`, `(0, 1)`, `(0,2)`…; those for $ξ$ would be similar, but starting with `1` instead of `0`.
# - A single high entropy value (`entropy`) is mixed with each key to create high-quality RNG seeds.
# - Those seeds are used to create independent RNGs.
#
# Moreover, to avoid accidentally breaking statistical independence, by default we prevent the same key from being used twice. In some cases this might actually be desired (e.g. to recreate earlier data), so a flag is provided to disable this check.
#
# Most of the heavy lifting here is done by `np.random.SeedSequence`, namely the conversion of low-entropy key tuples and the `entropy` value into high quality seeds for non-overlapping RNGs. This module adds:
#
# - A hard-coded `entropy` variable, which should be changed for each project.
# - Convenience functions `get_np_rng` and `get_shim_rng`, which take a key tuple and return an RNG.
#   + `get_np_rng` always returns a NumPy Generator object. Use `get_shim_rng` for an RNG that can be used in Theano expressions.
# - Verification (unless disabled) that each RNG key is used only once.
# - The function `draw_model_sample` to draw a reproducible sample from a PyMC3 model by providing an RNG key tuple.

# %%
from typing import Tuple, Optional, List
import numpy as np

# %%
from numpy.random import SeedSequence

# %% [markdown]
# ## Entropy value
#
# To ensure that no statistical correlations are introduced _projects_, the value `entropy` should be changed for each new project. To obtain a new value, simply execute the following in a code cell:
#
# ```python
# SeedSequence().entropy
# ```

# %%
entropy = 192010274972348534620754835903872732883

# %% [markdown]
# ## Seed generator

# %%
mother_seedseq = SeedSequence(entropy)
seedseqs = {}  # I'm not 100% sure that storing this outside a task is safe,
               # even if it is managed specifically for reproducibility.
seedseqs[()] = mother_seedseq
consumed_seed_keys = []

# %% [markdown]
# REMARK: The same mother seedsequence spawns all the RNG keys, so they can be
#    assumed to be independent.

# %%
# TODO?: Single-run task (could be used for CreateRNG)
def get_seedsequence(seed_key: Union[int, Tuple[int,...]],
                     exists_ok: bool=True) -> SeedSequence:
    """
    .. Important:: To ensure reproducibility, RNGs used within a task must not
       be used anywhere else (neither within another task, nor in module-level
       code). This is why, by default, an error is raised if a key is re-used.
       You can disable this check by passing the argument ``exists_ok=True``.
       A good rule of thumb for whether this is appropriate is that there should
       be only one line of code which uses a particular key. If that line is
       called more than once, than it is desirable, and probably OK, to set
       `exists_ok` to ``True``.

    .. todo:: A mechanism for tracking whether repeated calls with the same key
       come from the same code location (OK) or different locations (not OK).

    :param seed_key: The `spawn_key` of the returned SeedSequence.
    :param exists_ok: Whether it is ok to return a SeedSequence that was already
        generated. The functions below set the default to False to guarantee
        independence of RNGs.
        Only set to True if you are sure that RNGs sharing the same seed are
        not used together.
        Rule of thumb: it is safe to set `exists_ok` to ``True`` if the task
        which depends on the returned RNG, **as well as all upstream tasks**,
        is **neither recorded nor memoized**.
    """
    if not isinstance(seed_key, tuple):
        seed_key = (seed_key,)
    if not exists_ok and seed_key in consumed_seed_keys:
        # The problem here is the following: Suppose two recorded tasks A and B
        # both create their RNG with the same seed key.
        # The call to CreateRNG in B would return the *same RNG instance* as in
        # A, since task calls are cached. But then, although the tasks A and B
        # are disconnected, their outputs would depend on which was executed
        # first.
        # Alternatively, if we return different instances, they would produce
        # exactly identical streams of random numbers. There may be a use case
        # for this, but given the higher likelihood that it is a subtle, hard
        # to find error, I rather raise an exception. If I have a concrete
        # example, I can see how it might be integrated.
        raise RuntimeError(
            f"Attempted to create multiple RNGs with the same seed key "
            f"{seed_key}. This makes it impossible to ensure consistent "
            "reproducible tasks.")
    # Assert that there are no keys which use this one as parent
    # Even when `exists_ok` is True, we still don't allow for which the parent
    # is also used. I.e. if (3, 2, 1) is a used key, (3,) and (3, 2) are disallowed. (As well as (3,2,1,1), (3,2,1,2,), ...)
    assert not any(key[:len(seed_key)-1] == seed_key for key in consumed_seed_keys)
    # Loop over the key levels, creating all the needed parents
    for i in range(len(seed_key)):
        if seed_key[:i+1] not in seedseqs:
            # Calculate the number of existing keys with the same prefix
            n_existing_keys = len(list(
                filter(lambda k: len(k)==i+1 and k[:i]==seed_key[:i],
                       seedseqs)))
            n_spawns = seed_key[i] - n_existing_keys + 1
                # +1 because 0 is a key
            # Spawn as many seeds as needed to get to seed_key
            for seedseq in seedseqs[seed_key[:i]].spawn(n_spawns):
                assert len(seedseq.spawn_key) == i + 1
                assert seedseq.spawn_key not in seedseqs
                seedseqs[seedseq.spawn_key] = seedseq
    consumed_seed_keys.append(seed_key)
    return seedseqs[seed_key]

# %% [markdown]
# ## NumPy RNGs

# %%
def get_np_rng(seed_key: Union[Tuple[int,...], int], exists_ok: bool=False) -> RNGenerator:
    seedseq = get_seedsequence(seed_key, exists_ok)
    return np.random.Generator(np.random.PCG64(seedseq))

# %% [markdown]
# ## Theano RNGs
#
# The function `get_shim_rng` is almost identical to `get_np_rng`, but will returns a Theano RNG instead of a NumPy Generator when Theano is loaded through _theano_shim_.

# %%
import theano_shim as shim

def get_shim_rng(seed_key: Union[Tuple[int,...], int], exists_ok: bool=False) -> AnyRNG:
    seedseq = get_seedsequence(seed_key, exists_ok)
    rng = shim.config.RandomStream(seedseq.generate_state(1)[0])
    rng.name = f"RNG (seed: {seed_key})"
    return rng


# %% [markdown]
# ## PyMC3 sampler

# %%
import pymc3 as pm

def draw_model_sample(model: 'sinn.Model', key: Tuple[int],
                      var_names: Optional[List[str]]=None, n :int=1):
    """
    Draw `n` samples from the model.
    Wrapper around `pymc3.sample_prior_predictive` which ensures it is
    reproducible.
    """
    # This is essentially the Prior.random() method, distilled for generic
    # PyMC3 models.
    if var_names is None:
        var_names = [v.name for v in model.vars]
        # vars = free_RVs = all RVs which are not observed and not deterministics
    seed = get_seedsequence(key, exists_ok=True).generate_state(1)[0]
        # exists_ok=True is safe if `key` is not used elsewhere in the code
    var_names = sorted(var_names)
    return pm.sample_prior_predictive(
        samples=1,
        random_seed=seed,
        var_names=var_names,
        model=model)
# %%
