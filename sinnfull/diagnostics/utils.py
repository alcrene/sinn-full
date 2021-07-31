from __future__ import annotations

from warnings import warn
from types import SimpleNamespace
from typing import List, Sequence
import numpy as np
import theano_shim as shim
import sinn
import sinn.utils
from sinn.utils.pydantic import initializer
import smttask
import sinnfull
from sinnfull.viz import RSView, FitData
import sinnfull.diagnostics

rsview = RSView()
# NOTE: `load_record` -> RecordData
record = "Call `load_record` before other functions in this module."
task = "Call `load_record` before other functions in this module."
latents = "Call `load_record` before other functions in this module."
Θ = "Call `load_record` before other functions in this module."
optimizer = "Call `load_record` before other functions in this module."
model = "Call `load_record` before other functions in this module."
data = "Call `load_record` before other functions in this module."
λη = "Call `load_record` before other functions in this module."
ηhist = "Call `load_record` before other functions in this module."
    # The history to investigate (in particular, the one for which λη is calculated)

# TODO: Change to the functions below

class RecordData:
    """
    Wraps a FitData objects and adds introspection attributes/methods.

    .. Note:: Needs to be updated; see `set_to_step`. We should be able to do
       without globabl variables.
    """
    record: RecordView
    task: Task
    logp: LogpRecorder
    Θ: ΘRecorder
    latents: LatentsRecorder
    model: sinn.Model
    data: Array
    λη: float
    full_gradη: Callable
    sliced_gradη: Callable

    def __init__(self, record_label, ηhist:str):
        """
        :param:λη_hists: If provided, latent gradients are only computed for these
            histories. Default is to compute all
        """
        ## Basics ##
        # global task, latents, optimizer, model, data
        record = rsview.get(record_label)
        task = smttask.Task.from_desc(record.parameters)
        optimizer = task.optimizer.run()
        # Replace the optimizer in `record` so that FitData can use its
        # attributes, in particular its instantiated Model.
        # Otherwise fit.model and optimizer.model will be two separate instances
        record.parameters.inputs.optimizer = optimizer
        globals()['ηhist'] = ηhist
        # record_data = SimpleNamespace(
        #     record = record,
        #     task = task,
        #     logp = record.get_output('log L'),
        #     latents = record.get_output('latents'),
        #     Θ = record.get_output('Θ'),
        #     optimizer = optimizer,
        #     model = optimizer.model,
        #     data = next(iter(optimizer.data_segments))[2],
        #     λη = shim.eval(optimizer.fit_hyperparams['latents']['λη'][ηhist])
        # )
        self.fit = FitData(record=record, θspace='optim')
            # In order to continue fits, we need to load parameters in
            # optim space
        self.task = task
        self.optimizer = optimizer
        self.ηhist = getattr(optimizer.model, ηhist)
        self.data = next(iter(optimizer.data_segments))[2]
        self.λη = shim.eval(optimizer.fit_hyperparams['latents']['λη'][ηhist])
        # # Following should be done by Pydantic
        # from mackelab_toolbox.typing import Array
        # record_data.latents.values = [[Array['float64'].validate(v) for v in vallist]
        #                               for vallist in record_data.latents.values]

        ## Inspection of gradients ##
        global full_gradη, sliced_gradη
        full_gradη, sliced_gradη = sinnfull.optim.get_gradη(
            optimizer, upd_type='default', wrt_hist=getattr(optimizer.model,ηhist),
            full_gradη=True, sliced_gradη=True)
        self.full_gradη = full_gradη
        self.sliced_gradη = sliced_gradη

        # Update globals – Eventually this will be deprecated
        g_dict = globals()
        for k, v in g_dict.items():
            g_dict[k] = getattr(self, k, v)
        # globals().update(self.__dict__)
        # return record_data

    @property
    def record(self):
        return self.fit.record
    @property
    def logp(self):
        return self.fit.logL_evol
    @property
    def Θ(self):
        return self.fit.Θ_evol
    @property
    def latents(self):
        return self.fit.latents_evol
    @property
    def model(self):
        return self.fit.model
    @property
    def Θprior(self):
        return self.optimizer.prior_params

    ## Moving across the fit iterations ##

    def set_to_step(self, step, verbose=True):
        """
        Set the model to one of the inference steps.
        This will look for the nearest step `ηstep` at which the latents were
        recorded, and then nearest step `θstep` at which parameters were recorded.
        The model is then set to have the latents at `ηstep` and the parameters
        at `θstep`.

        .. Warning:: The module-scode `step_to_step` function below is much
           improved compared to this one and should be used instead.

        .. Note:: Unless the Θ and latent recorders are synchronized such that there
           is an `θstep` matching each `ηstep`, the reconstructed state may differ
           slightly from the one which occurred during training.
        """
        ηstep = self.latents.get_nearest_step(step)
        θstep = self.Θ.get_nearest_step(ηstep)
        model = self.model
        if verbose:
            print(f"Setting to optimization step {ηstep}.")
        if ηstep != θstep:
            warn("ηstep and θstep differ; reconstructed state will differ from "
                 "the state which occurred during the optimization.\n"
                 f"ηstep: {ηstep}\tθstep: {θstep}")
        with sinn.utils.unlocked_hists(*model.history_set):
            for h in self.optimizer.latent_hists:
                h[:] = self.latents[h.name, ηstep]
            for h in self.optimizer.observed_hists:
                h[:] = self.data[h.name].data
        # NB: I'm not sure how useful it is to set the model parameters,
        #     since it is the set optimizer.Θ that is optimized
        model.params.set_values(
            self.Θprior.backward_transform_params(Θ[θstep]))
        self.optimizer.Θ.set_values(Θ[θstep])
        if hasattr(model, 'remove_degeneracies'):
            model.remove_degeneracies()
        # Set the optimizer step
        self.optimizer.stepi = ηstep

def set_to_zero():
    """
    Zero the model histories.
    Model parameters are left unchanged
    """
    with sinn.utils.unlocked_hists(*model.history_set):
        for h in model.history_set:
            h[:] = 0


## Changing the state of an optimizer ##
# Recreate the state of an optimizer based on recorded state

from warnings import warn
from types import SimpleNamespace
from typing import Union, Dict
import xarray as xr
import sinn
from sinn.utils import unlocked_hists
from sinnfull.utils import dataset_from_histories
from sinn.utils import unlocked_hists
from sinnfull.utils import shift_time_t0

DataType = Union[Dict[str,Union[sinn.History,np.ndarray]], xr.Dataset]
def merge_data(timeaxis: sinn.TimeAxis, data: Union[DataType, Sequence[DataType]]
    ) -> xr.Dataset:
    """
    Combine time series data (either `~sinn.History` objects or plain NumPy
    arrays) into an xarray Dataset.
    Plain arrays are left-aligned with `timeaxis`.
    Time axes of given histories should align (have the same t0 and time steps)
    as `timeaxis`; they can however contain addition left padding and have a
    different length.
    """

    if isinstance(data, xr.Dataset):
        pass

    elif isinstance(data, Sequence):
        if any(isinstance(d, Sequence) for d in data):
            warn("While `merge_data` will work with nested lists, it is more "
                 "efficient to flatten them first.")
        data = xr.merge([merge_data(timeaxis, d) for d in data])

    elif isinstance(data, dict):
        # `data` is just a dict; we need to figure out the time length
        # by inspecting the data
        hist_items = {hnm: h for hnm, h in data.items()
                      if isinstance(h, sinn.History)}
        array_items = {hnm: d for hnm, d in data.items()
                       if isinstance(d, np.ndarray)}
        if len(hist_items) + len(array_items) != len(data):
            received_types = set(type(d) for d in data.values())
            raise TypeError("`data` elements must be either Histories or "
                            "ndarrays. Received elements of the following "
                            f"types: {received_types}.")

        # Convert NumPy arrays to xarray DataArrays
        data_arrays = []
        for hnm, hdata in array_items.items():
            coords = {'time': timeaxis.stops_array[:len(hdata)]}
            data_arrays.append(xr.DataArray(
                name=hnm,
                data=hdata,
                coords={'time': timeaxis.stops_array[:len(hdata)]},
                dims=['time'] + [f'{hnm}_{i}' for i in range(1, hdata.ndim)]
            ))

        # Construct `data` by merging DataArrays with a DataSet
        # from the histories.
        if hist_items:
            data = dataset_from_histories(
                hist_items.values(), names=hist_items.keys())
            if data_arrays:
                data = xr.merge([data, *data_arrays])
        else:
            data = xr.merge(data_arrays)

    else:
        raise TypeError("`data` must either be an xr.Dataset, a dictionary "
                        "of sinn.Histories and/or numpy arrays, or a list "
                        "of these types. Instead received a value of type "
                        f"{type(data)}.")

    # At this point we can assume data to be xr.Dataset

    # Remove bins from the end if not all histories have a value
    # NB: It's normal to have unequal *left* bin limits, because of padding.
    # Assumption: forward time, causal model.
    # We can't use `dropna`, because we want to keep the left padding
    time_idcs_to_drop = []
    for tidx in range(len(data.time)-1, -1, -1):
        if any(any(v) for v in np.isnan(data.isel(time=tidx)).data_vars.values()):
            time_idcs_to_drop.append(tidx)
        else:
            break
    if time_idcs_to_drop:
        data = data.drop_isel(time=time_idcs_to_drop)

    # Return data
    return data

def set_model(model,
              data: Union[Dict[str,Union[sinn.History,np.ndarray]],
                          xr.Dataset],
              params: Union[SimpleNamespace, dict]):
    """
    Set the model to a particular state, with give history data
    and parameter values.
    """
    ## Input normalization ##
    data = merge_data(model.time, data)

    # From this point we can assume that `data` is an xr.Dataset

    ## Set history values ##
    max_pad_left = max(h.pad_left for hnm, h in model.nested_histories.items()
                       if hnm in data.data_vars.keys())
        # We need this much of the initial data points to initialize
        # (Only consider histories in the provided data)
    missing_init = [hnm for hnm, h in model.nested_histories.items()
                    if not h.locked and h.pad_left and hnm not in data.data_vars.keys()]
    if missing_init:
        raise ValueError("Initialization data must be provided for unlocked "
                         "histories if their padding is > 0. Data for the "
                         f"following histories is missing: {sorted(missing_init)}.")
    K_padded = model.time.Index.Delta(len(data.time))
    K = K_padded - max_pad_left
        # The unpadded number of time bins
    # Clear the model, so that any unset histories are recomputed
    model.clear()

    ## Set parameter values ##
    # (This sets cur_tidx back to -1, so do this before hist updates)
    model.update_params(params)

    ## Ensure model and date time axes are aligned ##
    data_dt = data.time.attrs.get('dt', None)
    if data_dt is None:
        data_dt = np.diff(data.time).mean() * model.time.unit
    assert np.isclose(model.dt, data_dt)
    t0 = model.time.t0
    t0idx = data.time.searchsorted(t0)
        # Because `data` may include padding bins, and model.time typically
        # does not, t0 in the data and in model.time may not be the same.
    if not np.isclose(data.time[t0idx].data*model.time.unit, t0):
        # We can end up here if the model time axis wasn't shifted to align
        # with the data
        t0 = data.time[t0idx].data*model.time.unit
    if t0idx < max_pad_left:
        # Model t0 doesn't allow enough time bins for padding
        Δ = max_pad_left - t0idx
        t0 += Δ.plain*model.dt
    if not np.isclose(t0, model.t0):
        # We had to shift the model's t0, either because of lacking time bins,
        # or because it doesn't align with the data.
        t0idx = data.time.searchsorted(t0)
        assert np.isclose(data.time[t0idx].data*model.time.unit, t0)
        shift_time_t0(model, t0)
        # shift_time_t0(model.time, t0)
        # for m in model.nested_models.values():
        #     shift_time_t0(m.time, t0)
        # for h in model.history_set:
        #     shift_time_t0(h.time, t0)
        
    ## Set all histories to the values provided by `data` ##
    for hname, hdata in data.data_vars.items():
        h = getattr(model, hname)
        with unlocked_hists(h):
            h[:h.t0idx+K] = hdata[t0idx-h.pad_left:t0idx+K]
                # NB: By using AxisIndex objects here, we make use of
                # AxisIndex's validation to ensure each history uses
                # the right padding and goes exactly up to K.
                # For the difference between axis and data indices, see
                # https://sinn.readthedocs.io/en/latest/axis.html#indexing

    if hasattr(model, 'remove_degeneracies'):
        model.remove_degeneracies()

def set_to_step(optimizer, fitdata, step, verbose=True):
    """
    Set the model to one of the inference steps.
    This will look for the nearest step `ηstep` (relative to `step`) at which
    the latents were recorded, and then the nearest step `θstep` (relative to
    `ηstep`) at which parameters were recorded.
    The model is then set to have the latents at `ηstep` and the parameters
    at `θstep`.

    .. Note: Unless the Θ and latent recorders are synchronized such that there
       is an `θstep` matching each `ηstep`, the reconstructed state may differ
       slightly from the one which occurred during training.
    """
    if fitdata.Θspace != 'optim':
        raise ValueError("The provided FitData was not loaded with the option "
                         "`Θspace='optim'`")
    if step == -1:
        step = fitdata.latents_evol.steps[-1]
    ηstep = fitdata.latents_evol.get_nearest_step(step)
    θstep = fitdata.Θ_evol.get_nearest_step(ηstep)
    ηstepidx = fitdata.latents_evol.steps.index(ηstep)
    if verbose:
        print(f"Setting to optimization step {ηstep}.")
    if ηstep != θstep:
        warn("ηstep and θstep differ; reconstructed state will differ from "
             "the state which occurred during the optimization.\n"
             f"ηstep: {ηstep}\tθstep: {θstep}")

    # Recover data at step ηstep -- copied from FitData.η_curves
    # TODO: Use fitdata.get_observed_data
    model = optimizer.model
    if hasattr(fitdata.latents_evol, 'segment_keys'):
        segmentkey = fitdata.latents_evol.segment_keys[ηstepidx]
    else:
        segmentkey = None
    if segmentkey:
        *trialkey, t0, stop, tstep = segmentkey
        dt = model.time.dt; dt = getattr(dt, 'magnitude', dt)
        # Find the number of padding time bins we need to initialize the observed hists
        max_pad_left = max(h.pad_left for h in optimizer.observed_hists.values())
        # Shift time arrays so they start with the same value as the data
        t0_minus_pad = t0
        t0 = t0 + max_pad_left * dt
            # When the optimizer draws data segments, it reserves this `max_pad_left`
            # time bins to set the initial conditions (see `Optimizer.draw_data_segment`)
        if tstep:
            assert np.isclose(dt, tstep)
        shift_time_t0(model, t0)
        # shift_time_t0(model.time, t0)
        # for m in model.nested_models.values():
        #     shift_time_t0(m.time, t0)
        # for h in model.history_set:
        #     shift_time_t0(h.time, t0)
        # Retrieve the observed data for the corresponding slice
        observed_data = (fitdata.data_accessor.load(tuple(trialkey))
                         [[hnm for hnm in optimizer.observed_hists]]  # Keep only observed histories
                         .sel(time=slice(t0_minus_pad, stop)))  # Index the corresponding time, adding enough padding so that the data's t0 is actually t0
                            # NB: multiplying dt like this is numerically fragile, but should be fine for small Δ (< 100 or so)

    latents = fitdata.latents_evol[ηstep]

    # If parameters are in optim space, transform them to model space
    Θvals = fitdata.Θ_evol[θstep]
    assert set(Θvals) <= set(fitdata.prior.optim_vars)
    Θvals_model = fitdata.prior.backward_transform_params(Θvals)

    # TODO: Move to a function so these are are also applied with `set_model`
    # Set the optimizer's model
    set_model(optimizer.model,
              data=[latents, observed_data],
              params=Θvals_model)
    K = optimizer.model.cur_tidx - optimizer.model.t0idx + 1
    optimizer.Kshared.set_value(K.plain)
    # Set the optimizer's parameters
    optimizer.Θ.set_values(Θvals)
    # Set the optimizer step
    optimizer.stepi = ηstep

## Inspection of parameters ##

# TODO: Relevant functionality should be moved to GradInspector

from sinnfull.viz.config import pretty_names
from sinnfull.viz.typing_ import StrTuple
import holoviews as hv

def param_val_tables(Θ: dict) -> hv.Layout:
    """
    Return a layout composed of one table per parameter.

    .. Note:: For brevity, any qualifier is removed from parameter names; this
       means that a transformed variable may be labeled with its untransformed
       label. In the case of transformed variables in particular, this can
       lead to values outside of their expected domain.

    :param:Θ: Parameter dictionary.
    """
    from holoviews import Store
    Store.add_style_opts(hv.Table, ['sizing_mode'], backend='bokeh')
        # To do this outside the function would require calling `hv.extension('bokeh')` before importing this module

    tbs = {}
    for θnm, θval in Θ.items():
        assert θval.ndim <= 2
        θval = np.atleast_2d(θval)
        nrows, ncols = θval.shape
        θlbl = pretty_names.unicode[θnm.strip('_').split('_', 1)[0]]
        tbs[θnm] = hv.Table(θval, [f'{θlbl}{i}'.format(i) for i in range(ncols)],
                            # Don't assign label – it is forcibly 400px wide
                           ).opts(width=40+60*ncols, height=25*(nrows+1))
            # 40: width of index column

    layout = hv.Layout(list(tbs.values()))
    layout.opts(hv.opts.Table(sizing_mode="fixed"))

    return layout

## Inspection of gradients ##
full_gradη = "Call `set_record` before other functions in this module."
sliced_gradη = "Call `set_record` before other functions in this module."

def print_gradient(b, Kηb, Kηr):
    gη = full_gradη(b, Kηb, Kηr)
    cw = max(len(f"{g:.8f}") for g in gη.flat)
    w = len(gη[0])*(cw+1) + 1
    print(f"{'k':^5}" + "  " + f"{'-λη*gη_k':^{w}}")
    print("-"*(5+2+w))
    for k in range(b-1, min(b+Kηb+Kηr+3, len(gη))):  # HACK: +1 b/c hist padding
        gη_str = "(" + ", ".join(f"{gηki: >{cw}.8f}" for gηki in -λη*gη[k]) + ")"
        print(f"{k-getattr(model,ηhist).pad_left:>4d}  {gη_str}")

def print_data(b, Kηb, Kηr):
    data = getattr(model,ηhist)._num_data.get_value()  # Get the raw data w/ padding, so it aligns with gη
    cw = max(len(f"{g:.5f}") for g in data.flat)
    w = len(data[0])*(cw+1) + 1
    print(f"{'k':^5}" + "  " + f"{ηhist+'_k'}:^{w}")
    print("-"*(5+2+w))
    for k in range(b-1, min(b+Kηb+Kηr+3, len(data))):  # HACK: +1 b/c hist padding
        datak_str = "(" + ", ".join(f"{dataki: >{cw}.5f}" for dataki in data[k]) + ")"
        print(f"{k-getattr(model,ηhist).pad_left:>4d}  {datak_str}")

def parse_theano_print_output(s, filter_str=None):
    """Construct array from multiple lines of printed Theano outputs

    This is absolutely a hack, and the result should be checked for correctness.
    FIXME: At present, each line will be flattened in the output.
    """
    lines = s.split('\n')
    if filter:
        lines = [line for line in lines if filter_str in line]
    arr1D = np.fromstring(" ".join(line[:line.find('__str__')]
                                   for line in lines)
                          .replace('[', '')
                          .replace(']', ''),
                          sep=' ')
    return arr1D.reshape(len(lines), -1)


from typing import Union, Optional, List
from dataclasses import dataclass
from sinnfull.optim import DiagnosticRecorder

@dataclass
class single_update_record_condition:
    """
    For each step, record only as many times as there are elements in `target_b`.
    Will record the first time b_s is equal or lesser than one of the values in `target_b`.
    """
    target_b: List[int]
    # Internal
    last_step: Optional[int]=None
    last_b  : Union[int, float]=np.inf

    def __call__(self, stepi, b, ctx):
        if 'η' not in ctx:
            return False
        if stepi != self.last_step:
            self.last_b = np.inf
        available_b = [b for b in self.target_b if b < self.last_b]
        if available_b and b <= max(available_b):
            self.last_step = stepi
            self.last_b = b
            return True
        else:
            return False

    def clear(self):
        self.last_step = None
        self.last_b = np.inf

class PartialLogpRecorder(DiagnosticRecorder):
    """Compute the partial batch logp from k to end for different k."""
    name: str = 'partial_logp'
    start_k: List[int]
    partial_lengths: List[int]

    @initializer('callback')
    def make_partial_logp_callback(cls, callback, start_k, partial_lengths):
        k_list = start_k
        K_list = partial_lengths
        def partial_logp(optimizer):
            return [optimizer.logp(k, K) for k, K in zip(k_list, K_list)]
        return partial_logp

    @initializer('keys')
    def set_keys(cls, keys, partial_lengths):
        return [str(K) for K in partial_lengths]

    @initializer('record_condition')
    def set_record_condition(cls, record_condition, start_k):
        return single_update_record_condition(target_b=start_k)

class GradInspector:

    rec_data: RecordData
    η_gt: hv.HoloMap  # Ground truth curve

    # If needed, we can explicitely specify which fields to include in the tooltip this way:
    # from bokeh.models import HoverTool
    # hover = HoverTool(tooltips=[("grad η_i", "$grad")])

    def __init__(self, rec_data):
        self.rec_data = rec_data

        # Get ground truth curves
        η_gt = rec_data.fit.ground_truth_η_curves().select(hist=ηhist.name).collate(drop_constant=True)
        η_gt.opts(framewise=True);
        η_gt = hv.HoloMap({k: η_gt[k].to.area() for k in η_gt.keys()},
                          kdims=η_gt.kdims)
        self.η_gt = η_gt

    def ηgrad_plot(self, step, hist_idx, cmap='bjy') -> hv.Overlay:
        """
        Plot component `hist_idx` of the inferred and true latent history at step `step`.
        True latent is shown as a shaded area, arbitrarily shaded from 0.
        Inferred latent is shown as a color-coded curve, where color indicates the
        gradient of the likelihood with respect to that point, and thus an estimate
        of how it will change in the next iteration.

        .. Note:: Because updates to the latent are done in random batches, the actual
           change between two steps will differ from that estimated by the gradient.
           In particular, it will depend on the number of batches wich include that
           time point between two iteration steps.
        """
        #NB: Commented-out lines are useful for returning a HoloMap

        hist_idx = int(hist_idx)

        rec_data = self.rec_data
        ηhist = rec_data.ηhist
        η_gt = self.η_gt

        # Set to the specified step
        rec_data.set_to_step(step, verbose=False)

        # Retrieve gradient at this step
        K = len(rec_data.data.time) # (excludes padding for init. condition)
        η_grad = rec_data.full_gradη(0, K, 0)
            # Arguments: start k, batch length, relax window length
            # Only the sum (batch_length + window_len) matters for full_gradη
            # Has length (K + init_cond_padding)

        # TODO?: Rename to η_upd ?
        η_grad  = -rec_data.λη*η_grad
        # **HACK:** Remove the initial condition because it's not currently included in ground truth
        η_grad = η_grad[-K:]

        # Get the latent history at this step


    #    idcs = [(StrTuple(idx), idx) for idx in np.ndindex(ηhist.shape)]
    #    η = hv.HoloMap({idx: hv.Curve(zip(ηhist.time_stops, ηhist.get_data_trace(*idx_tup)),
    #                                  kdims=η_gt[idx].kdims, vdims=η_gt[idx].vdims)
    #                                   #    kdims=['time'], vdims=[ηhist.name+"_"+str(idx).replace("(", "").replace(")", "").replace(",", "")])
    #                    for idx, idx_tup in idcs}, kdims=η_gt.kdims)
    #    η.opts(framewise=True);
        idx = StrTuple((hist_idx,))
        ηi = hv.Curve(zip(ηhist.time_stops, ηhist.get_data_trace(hist_idx)),
                          kdims=η_gt[idx].kdims, vdims=η_gt[idx].vdims)

        # For each latent component, convert it to a path & concatenate with the gradient

    #    ηpaths = {}
        assert len(ηhist.shape) == 1
    #    for i in range(ηhist.shape[0]):
    #    ηi = η[f'({i},)']
        path_data = np.concatenate((ηi.data.to_numpy(), η_grad[:,hist_idx:hist_idx+1]), axis=1)
        ηpath = hv.Path(path_data, kdims=ηi.kdims+ηi.vdims, vdims=['grad'])
        ηpath.opts(line_width=1.5, color='grad', cmap=cmap, colorbar=True,)
    #    ηpaths[StrTuple((i,))] = ηpath
    #    η_grads = hv.HoloMap(ηpaths, kdims=η_gt.kdims)
    #    η_grads.opts(framewise=True);

    #    ov = η_gt * η_grads
        ov = η_gt[idx] * ηpath
        ov.opts(hv.opts.Area(fill_alpha=0.2, color="#888888", line_alpha=0),
                # hv.opts.Path(tools=['hover']) # Currently broken; see https://github.com/holoviz/holoviews/issues/4862
               )

        return ov
