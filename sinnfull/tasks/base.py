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
#     display_name: Python (comp)
#     language: python
#     name: comp
# ---

# %% [markdown]
# # Base tasks
#
# For creating data accessors, creating models, and performing optimization.

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('theano')

# %% tags=["hide-input"]
import os
import logging
from warnings import warn
from typing import (
    Union, Optional, Callable, Generator, Iterable, Sequence, List, Tuple, Dict)
from pathlib import Path  # Only used for typing

import theano_shim as shim
import numpy as np

# %% tags=["hide-input"]
import smttask
from smttask import (Task, TaskOutput, RecordedTask, RecordedIterativeTask,
                     MemoizedTask, NonMemoizedTask, UnpureMemoizedTask)
from smttask.typing import separate_outputs, PureFunction

import mackelab_toolbox as mtb
from mackelab_toolbox.typing import (RNGenerator, AnyRNG, PintValue, Array)

# %% tags=["hide-input"]
from sinn.histories import History
from sinn.models import Model, TimeAxis

from sinnfull.typing_ import IndexableNamespace
from sinnfull.optim import AlternatedSGD, Recorder, OptimizerStatus, ConvergenceTest
from sinnfull.sampling import SegmentSampler, FixedSegmentSampler
from sinnfull.data import DataAccessor
from sinnfull.data.synthetic import SyntheticDataAccessor
from sinnfull.rng import get_seedsequence, get_np_rng, get_shim_rng, draw_model_sample
from sinnfull.models.base import AccumulatedObjectiveFunction, Prior
from sinnfull.optim import OptimParams, Optimizer

# %% tags=["remove-cell"]
logger = logging.getLogger(__file__)

# %%
__all__ = ['CreateSyntheticDataset', 'CreateFixedSegmentSampler',
           'CreateModel', 'CreateOptimizer', 'OptimizeModel']


# %% [markdown]
# ## Data tasks

# %% [markdown]
# ### `CreateSyntheticDataset`

# %% tags=["hide-input"]
@MemoizedTask(json_encoders=sinnfull.json_encoders)
def CreateSyntheticDataset(
    projectdir: Union[str,Path],
    model_name: str,
    time: TimeAxis,
    prior: Prior,
    param_keys: List[Tuple[int,...]],
    sim_keys: List[Tuple[int,...]],
    init_conds: Optional[Union[Dict[str,Array], List[Dict[str,Array]]]] = None
) -> SyntheticDataAccessor:
    """
    Create a data set by simulating a model.

    Parameters
    ----------
    model_name: Name of the model which will be simulated to generate the data.
        Must match one of the models in ~sinnfull.models.
    time: Defines the length and time step of a simulation.
    prior: Model parameters will be sampled from this object, given `param_keys`.
        Must correspond to the model set by `model_name`.
    param_keys: Each value should be an RNG key tuple (see `sinnfull.rng`);
        Each different key tuple defines a different set of parameters, obtained
        by sampling `prior`.
    sim_keys: Each value should be an RNG key tuple, and there must be as many
        as there are `param_keys`. These keys are used to seed the model simulator.
    init_conds: By default, initial conditions are sampled from the model's
        stationary distribution, but they can also be specified explicitely
        as a list of dictionaries of {'variable name' <str>: initial value <Array>}
        pairs. This list must have the same length as `param_keys`.
        If single dictionary is provided, the same initial condition
        is used for all parameter sets.
    """
    # FIXME: Make API so we don't need to use private _load_types
    model_cls = mtb.iotools._load_types[model_name]
    # We use `param_key` to sample both the parameter and initial condition
    # In order to have consistent RNG state, we sample the IC immediately
    # after the parameters, with the same RNG object
    # (at present, PyMC3 still uses the default global RandomState)
    param_sets = []
    if init_conds is None:
        # Normal code path
        init_conds = []
    else:
        # Specialization for specifying ICs
        if isinstance(init_conds, dict):
            init_conds = [init_conds]*len(param_keys)
        else:
            assert isinstance(init_conds, list)
            if len(init_conds) != len(param_keys):
                if len(init_conds) == 1:
                    init_conds = init_conds*len(param_keys)
                else:
                    raise ValueError("When provided as a list, `init_conds` "
                                     "must have the same length as `param_keys`.")
        assert all(all(isinstance(v, np.array) for v in ic) for ic in init_conds)
    for key in param_keys:
        Θ = model_cls.Parameters(**prior.random(key))
        param_sets.append(Θ)
        if len(init_conds) < len(param_keys):
            # Normal code path
            model_stationary = model_cls.stationary_dist(params=Θ)
                # stationary_dist() returns a PyMC3 model
            # Use a key derived from `key` for the ICs
            init_conds.append(draw_model_sample(model_stationary, key=(*key,1)))
    param_sets = [model_cls.Parameters(**prior.random(key))
                  for key in param_keys]
    if hasattr(model_cls, 'remove_degeneracies'):
        param_sets = [model_cls.remove_degeneracies(Θ) for Θ in param_sets]
    # Just use the first key to create the RNG. It will be reseeded by the
    # DataAccessor anyway
    rng = get_shim_rng(sim_keys[0], exists_ok=True)  # exists_ok safe because we reseed at every draw
    model = model_cls(time=time, params=param_sets[0], rng=rng)
    seeds = [get_seedsequence(key, exists_ok=True).generate_state(1) for key in sim_keys]
    return SyntheticDataAccessor(
        projectdir, model=model, param_sets=param_sets, init_conds=init_conds, seeds=seeds)

# %% [markdown]
# ### `CreateFixedSegmentSampler`

# %% tags=["hide-input"]
@MemoizedTask(json_encoders=sinnfull.json_encoders)
def CreateFixedSegmentSampler(*,
    data        : DataAccessor,
    trial_filter: dict=None,
    rng_key     : Union[Tuple[int], int],
    t0          : PintValue,
    T           : PintValue
) -> FixedSegmentSampler:
    return FixedSegmentSampler(
        data=data, trial_filter=trial_filter, rng_key=rng_key,
        t0=t0, T=T)

# %% [markdown]
# ## Model tasks

# %% [markdown]
# ### `CreateModel`

# %% tags=["hide-input"]
# Don't memoize because the returned model is mutable => It could be changed
# by the calling code (or another task), and then the cached result would not
# be consistent.

# @NonMemoizedTask(json_encoders=sinnfull.json_encoders)
# def CreateModel(
#     model_class: str,
#     time: TimeAxis,
#     params: IndexableNamespace,
#     rng_key: Union[Tuple[int], int]
# ) -> Model:
#     ModelClass = mtb.iotools._load_types[model_class]
#     rng = get_shim_rng(rng_key, exists_ok=True)  # Not memoizing makes exists_ok safe
#     return ModelClass(time=time, params=ModelClass.Parameters(**params), rng=rng)

@NonMemoizedTask(json_encoders=sinnfull.json_encoders)
def CreateModel(
    time     : TimeAxis,
    model    : str,
    params   : IndexableNamespace,
    rng_key  : Optional[Union[Tuple[int,...], int]] = None,
    submodels: Optional[Dict[str,str]] = None,
    subparams: Optional[Dict[str,IndexableNamespace]] = None,
    connect  : Optional[Union[Dict[str,str], List[str]]] = None
) -> Model:
    """
    Create (possibly composite) models.

    Composite models are specified by providing a list of model class names
    and parameters.
    The `connect` argument is used to specify which histories from one submodel
    are used as inputs to another.

    .. Note:: Nested composite models are not currently supported.

    Parameters
    ----------
    ...
    models: The model to create, or a list of models to compose.
        When composing models, the order matters: going from left to right,
        in must be possible to instantiate the n-th model using only the
        histories of the preceding models as inputs.
        In other words, model dependencies (as specified by `connect`) must
        form a DAG, and the order of `models` should start with the root nodes
        of the DAG.
    params: Parameter values for the specified model(s).
        If `models` is a list, `params` must be a list of same length:
        the n-th parameter set will be used to instantiate the n-th model.
    rng_key: Any value accepted by `sinnfull.rng.get_shim_rng`.
    connect: (Optional – only valid for composite models)
        Dictionary of mappings from the `History` in one model to the `History`
        in another. Mapping should be in the direction
        `{lower_model.history_name: upper_model.history_name}`, where
        `lower_model` refers to the model which can be computed without the
        “upper” model. To avoid having to remember the key-value direction, an
        alternative notation is also accepted, where each pairing is specified
        as a single string: `"lower_model.history_name -> upper_model.history_name"`.
        These strings should then be passed in a list.
    """
    ## Validation & arg standardization
    # TODO: Can we move this into the Task validation, so that it is run
    #       during Task creation rather than execution ?
    if submodels:
        if not isinstance(submodels, dict) or not isinstance(subparams, dict):
            raise TypeError("Both `submodels` and `subparams` must be "
                            "dictionaries, with keys matching the attributes "
                            f"for the nested models in {model}.")
        elif len(subparams) != len(submodels):
            raise ValueError("There must be as many parameter sets as there "
                             f"are models.\nNo. models: {len(models)} "
                             f"({models})\nNo. parameter sets: {n_params}")
        elif set(subparams) != set(submodels):
            raise ValueError("The sets of submodel names and parameters don't "
                             "have the same keys:\n"
                             f"Submodel class name keys: {submodels.keys()}\n"
                             f"Submodel param set keys:  {subparams.keys()}")
        # TODO? Check that keys correspond to composite model attributes ?
        if connect is None:
            raise TypeError("`connect` argument is required to create a "
                            "composite model.")
    else:
        if connect is not None:
            raise TypeError("`connect` argument is only valid when creating a "
                            "composite model.")
    # Parse special List[str] syntax for `connect`
    if isinstance(connect, list):
        connect = {srchist.strip(): targethist.strip()
                   for srchist, targethist in
                   (cstr.split('->') for cstr in connect)}
    ## Identify the lower and upper model for each connection
    connect_by_hist = {submodel: [] for submodel in submodels}
    inverse_submodel_mapping = {v:k for k,v in submodels.items()}
        # Used to convert class names to attribute names in composite model
    # If the same class is used for more than one attribute, don't allow using
    # it, to avoid hard to track bugs.
    cls_names = list(submodels.values())
    for cls_nm in list(inverse_submodel_mapping.keys()):
        if cls_names.count(cls_names) > 1:
            del inverse_submodel_mapping[cls_nm]
    # Now build the connection dictionary
    for lowmodelhist, upmodelhist in connect.items():
        low_model, low_hist = lowmodelhist.split('.')
        up_model, up_hist = upmodelhist.split('.')
        # Allow models in `connect` to be specified either by their class name,
        # or the corresponding attribute of the composite model.
        # Since the same model class can in theory be used multiple times, we
        # give precedence to the attribute name, and use that for the
        # connection dictionary.
        if low_model not in connect_by_hist:
            low_model = inverse_submodel_mapping[low_model]
        if up_model not in connect_by_hist:
            up_model = inverse_submodel_mapping[up_model]
        connect_by_hist[up_model].append((low_model, low_hist, up_hist))
    # TODO: It would be nice to ensure all models connections form a DAG

    ## Task execution
    ModelClass = mtb.iotools._load_types[model]
    if rng_key is not None:
        rng = get_shim_rng(rng_key, exists_ok=True)
            # Not memoizing makes exists_ok safe, as long as we use the same
            # RNG instance for each submodel
    if submodels:
        submodel_names = submodels
        submodels = {}
        for submodel_attr in submodel_names.keys():
            submodel_clsname = submodel_names[submodel_attr]
            submodel_Θ = subparams[submodel_attr]
            SubmodelClass = mtb.iotools._load_types[submodel_clsname]
            connected_hists = {up_hist: getattr(submodels[low_model], low_hist)
                               for low_model, low_hist, up_hist
                               in connect_by_hist[submodel_attr]}
            sub_extra_args = {}
            # TODO: Check field type instead of name for RNG
            if 'rng' in SubmodelClass.__fields__:
                sub_extra_args['rng'] = rng
            submodels[submodel_attr] = SubmodelClass(
                time=time, params=SubmodelClass.Parameters(**submodel_Θ),
                **connected_hists, **sub_extra_args)
        extra_args = submodels
    else:
        extra_args = {}
    # TODO: Check field type instead of name, to allow different name for RNG
    if 'rng' in ModelClass.__fields__:
        extra_args['rng'] = rng
    return ModelClass(time=time, params=ModelClass.Parameters(**params),
                      **extra_args)


# %% [markdown]
# ## Optimization tasks

# %% [markdown]
# ### `CreateOptimizer`

# %% tags=["hide-input"]
@MemoizedTask(json_encoders=sinnfull.json_encoders)
def CreateOptimizer(
    model             : Model,
    rng_key           : Union[Tuple[int], int],
    data_segments     : SegmentSampler,
    observed_hists    : Sequence[Union[History,str]],
    latent_hists      : Sequence[Union[History,str]],
    # params            : dict,  # Type for gradient masks
    # params            : List[str],
    prior_params      : Prior,
    init_params       : OptimParams,
    fit_hyperparams   : dict,
    update_hyperparams: Optional[PureFunction[[AlternatedSGD],dict]],
    logp              : Optional[AccumulatedObjectiveFunction]=None,
    logp_nodyn        : Optional[AccumulatedObjectiveFunction]=None,
    logp_params       : Optional[AccumulatedObjectiveFunction]=None,
    logp_latents      : Optional[AccumulatedObjectiveFunction]=None,
    logp_latents_nodyn: Optional[AccumulatedObjectiveFunction]=None,
    param_optimizer   : Optional[PureFunction]=None,
    latent_optimizer  : Optional[PureFunction]=None,
    # latent_cache_path : str=".cache/latents",
    # convergence_tests : List[ConvergenceTest]=[]
) -> AlternatedSGD:
    """
    This task just forwards every argument to `AlternatedSGD`.
    The only thing it does before hand is convert `rng_key` to an RNG instance,
    and ensure that observed histories are locked.
    """
    rng = get_np_rng(rng_key)
    # latent_cache_path must not affect digest, so we hard-code it
    # If it is being launched by smttask, multiple tasks may run in parallel
    # – to avoid clashing caches, we append the smttask process number to the cache path
    latent_cache_path = ".cache/latents_process"
    if "SMTTASK_PROCESS_NUM" in os.environ:
        n = os.environ['SMTTASK_PROCESS_NUM']
        latent_cache_path += f"-{n}"
    optimizer = AlternatedSGD(
        model=model, rng=rng, data_segments=data_segments,
        observed_hists=observed_hists, latent_hists=latent_hists,
        prior_params=prior_params, init_params=init_params,
        fit_hyperparams=fit_hyperparams, update_hyperparams=update_hyperparams,
        logp=logp, logp_nodyn=logp_nodyn, logp_params=logp_params,
        logp_latents=logp_latents, logp_latents_nodyn=logp_latents_nodyn,
        # logp_params_regularizer=logp_params_regularizer,
        param_optimizer=param_optimizer, latent_optimizer=latent_optimizer,
        latent_cache_path=latent_cache_path)
    if shim.config.library != 'numpy':
        optimizer.compile_optimization_functions()
    else:
        warn("No symbolic library has been loaded through theano_shim: "
             "it will not be possible to use this optimizer.")
    for h in optimizer.observed_hists:
        h.lock(warn=False)
    return optimizer

# %% [markdown]
# ### `OptimizeModel`

# %% tags=["hide-input"]
class OptimizeOutputs(TaskOutput):
    nsteps   : int
    optimizer: AlternatedSGD
    recorders: separate_outputs(
        Recorder, lambda recorders: [rec.name if isinstance(rec, Recorder)
                                     else rec['name']
                                     for rec in recorders])
        # The lambda arguments must match names in the input signature
        # Lambda should work with both the expected type, and its dict serialization
    # logL     : Recorder
    # params   : Recorder
    # latents  : Recorder
@RecordedIterativeTask('nsteps', map={'optimizer': 'optimizer',
                                      'recorders': 'recorders'},
                       json_encoders=sinnfull.json_encoders)
def OptimizeModel(
    nsteps    : int,
    optimizer : Optimizer,
    recorders : Tuple[Recorder,...],
    convergence_tests : List[ConvergenceTest]=[],
    step_kwargs: dict={}
) -> OptimizeOutputs:
    """
    Parameters
    ----------
    nsteps: Number of optimizer steps to take
    optimizer: The optimizer to use. Must be a instance of `Optimizer`.
    recorders: A sequence of `Recorder` instances.
    step_kwargs: Additional keyword arguments to pass to `optimizer.step()`.
    """
    from tqdm.auto import tqdm
    import smttask
    from smttask.multiprocessing import get_worker_index

    # Abort fit if optimizer has already failed
    # This can happen if we restart from a previous fit.
    if optimizer.status is OptimizerStatus.Failed:
        outcome = f"Aborted: optimizer already failed at step {optimizer.stepi}."
        return {'nsteps':optimizer.stepi, 'optimizer':optimizer,
                'recorders': recorders, 'outcome': outcome}

    # Compile the optimizer
    if not optimizer.compiled:
        logger.info("Compiling optimization functions...")
        optimizer.compile_optimization_functions()
        logger.info("Done compiling.")
    # Attach recorders to optimizer
    for r in recorders:
        optimizer.add_recorder(r)
    # Record the initial state
    if optimizer.stepi == 0:
        # Make an initial estimate of the latents. This is a bit wasteful if
        # there are multiple segments (another segment will be drawn), but
        # it allows to record initial values for log L, latents,
        optimizer.draw_data_segment()
        optimizer.model.remove_degeneracies(exclude=optimizer.observed_hists)
        optimizer.record()
    # Run the optimizer for requested number of steps
    for i in tqdm(range(optimizer.stepi, nsteps),
                  desc=f"OptimizeModel {smttask.config.process_number}",
                  position=get_worker_index(),
                  leave=False  # When a fit is finished, free the space for the next. Helps keep progress bars tidy
                  ):
        optimizer.step(**step_kwargs)  # `step()` includes a call to `record()`
        # Record current state
        for recorder in optimizer.recorders.values():
            if force or recorder.ready(self.stepi):
                recorder.record(self.stepi, self)
        # Check if we reached an early stopping condition
        for convergence_test in convergence_tests:
            optimizer.status |= convergence_test(self)
        if (optimizer.status & OptimizerStatus.Converged) is OptimizerStatus.Converged:
            outcome = f"Terminated fit early with status <{optimizer.status}>."
            logger.info(outcome)
            optimizer.record()
            return {'nsteps': optimizer.stepi, 'optimizer': optimizer,
                    'recorders': recorders, 'outcome': outcome}
    # Record the final state
    optimizer.record()
    outcome = f"Terminated without converging. Status: <{optimizer.status}>."
    # Remove recorders from optimizer, so they aren't saved twice
    for r in recorders:
        optimizer.remove_recorder(r)
    # Return the results in the order prescribed by OptimizeOutputs
    return {'nsteps':optimizer.stepi, 'optimizer':optimizer,
            'recorders': recorders, 'outcome': outcome}
