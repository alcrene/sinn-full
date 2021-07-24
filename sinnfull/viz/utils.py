from mackelab_toolbox.utils import index_iter
from smttask import Task
from ._utils import *
    # interpolate_color
    # get_logL_quantile
    # convert_dynmap_layout_to_holomap_layout
from .config import pretty_names
from .record_store_viewer import FitData
from sinnfull.diagnostics.utils import set_model

## Retrieve record statistics ##

def get_ground_truth_logp(record) -> float:
    # TODO: There's a lot of unnecessary compilation when we reload the entire optimizer
    optimizer = Task.from_desc(record.get_param('optimizer')).run()
    fitdata = FitData(record=record, Θspace='model')
    set_model(optimizer.model,
              data=fitdata.ground_truth_η(),
              params=fitdata.ground_truth_Θ(split=False))
    for h in optimizer.model.history_set:
        h.lock()
    return optimizer.logp()

    # # The code below checks that the realization used to compute logp matches the
    # # one used for plotting
    # hmap = fitdata.ground_truth_η_curves().collate(drop_constant=True)
    # new_hmap = {}
    # for (hist_index, hist_name), gtη in hmap.items():
    #     hist = getattr(optimizer.model, hist_name)
    #     hist_index = literal_eval(hist_index)
    #     trace = hist.traces
    #     for α in hist_index:
    #         trace = trace[α]
    #     # Interpolate to the same time steps as the plotted data
    #     newdata = np.interp(gtη['time'], *zip(*trace))
    #     newη = hv.Curve(list(zip(gtη['time'], newdata)),
    #                     label="GT used for logp")
    #     newη.opts(alpha=0.7)
    #     ov = gtη.relabel("GT") * newη
    #     new_hmap[(hist_index, hist_name)] = ov.opts(show_legend=False)
    # ov.opts(show_legend=True)
    # layout = hv.HoloMap(new_hmap, kdims=hmap.kdims).layout().cols(2).opts(axiswise=True)
    # display(layout)
    
# ## Text & labels
#
# Return the dimension name from history and index

def get_histdim_name(hist: 'sinn.History', index: Tuple[int,...]) -> str:
    """
    Given a History and an index for a component of that history, return
    a standard string that can be used as a dimension name or identifiel.
    E.g.: given a history with the name 'rates' and the index (0, 1), returns
    'rates_01'.
    """
    return (f"{pretty_names.ascii.get(hist.name)}"
            f"{pretty_names.ascii.index_str(index, prefix='_')}")
    
def get_histdim_label(hist: 'sinn.History', index: Tuple[int,...], sep='') -> str:
    """
    Given a History and an index for a component of that history, return
    a standard string that can be used as a label.
    E.g.: given a history with the name 'rates' and the index (0, 1), returns
    'rates₀₁'.
    By default, multidimensional indices are concatenated with no separators;
    use `sep=','` if index values may go above 9.
    """
    return (f"{pretty_names.unicode.get(hist.name)}"
            f"{pretty_names.unicode.index_str(index)}")
            
def get_histdims(hists: List['sinn.History']) -> List[hv.Dimension]:
    return [hv.Dimension(get_histdim_name(h, idx),
                         label=get_histdim_label(h, idx))
            for h in hists for idx in index_iter(h.shape)]
