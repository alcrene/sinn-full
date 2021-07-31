from mackelab_toolbox.utils import index_iter
from ._utils import *
    # interpolate_color
    # get_logL_quantile
    # convert_dynmap_layout_to_holomap_layout
from .config import pretty_names

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
