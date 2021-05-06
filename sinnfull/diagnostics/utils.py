from __future__ import annotations

from warnings import warn
from types import SimpleNamespace
from typing import List
import numpy as np
import theano_shim as shim
import sinn
import sinn.utils
from sinn.utils.pydantic import initializer
import smttask
import sinnfull
from sinnfull.viewing import RSView, FitData
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

class RecordData:
    """
    Wraps a FitData objects and adds introspection attributes/methods.
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

        .. Note: Unless the Θ and latent recorders are synchronized such that there
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

## Inspection of parameters ##

# TODO: Relevant functionality should be moved to GradInspector

from sinnfull.viewing.config import pretty_names
from sinnfull.viewing.typing_ import StrTuple
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
