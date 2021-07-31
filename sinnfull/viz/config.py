# -*- coding: utf-8 -*-
# # Config
#
# Global configuration module for visualizations.

from __future__ import annotations
from typing import Union
from dataclasses import dataclass, field
from collections import defaultdict

# ------------------------------------------------
# ## Matplotlib styling
#
# `rcParams` is the name used by matplotlib for its config dictionary
#
# TODO: Move as much as possible to a style file

rcParams = {'font.sans-serif': ['CMU Bright', 'sans-serif'],
            'font.serif': ['Computer Modern Roman', 'serif'],
            'savefig.pad_inches': 0,                          # Don't change size when saving
            'figure.constrained_layout.use': True,            # Make sure there is space for labels
            'figure.figsize': [5.0, 3.0]
           }

# ------------------------------------------------
# ## Bokeh styling

import numpy as np
import matplotlib as mpl
import seaborn as sns
import holoviews as hv
from mackelab_toolbox.utils import Singleton
from sinnfull.viz._utils import interpolate_color

def default_factory(attr, f):
    def singleton_factory():
        try:
            return getattr(BokehOpts._Singleton__instance, attr)
        except AttributeError:
            # Only executes on first instantiation
            return f()
    return field(default_factory=singleton_factory)

@dataclass
class BokehOpts(metaclass=Singleton):
    row_height: int=25  # Default bokeh value: 25
    θ_curve_points = 75
    η_curve_points = 150

    def __new__(cls, *args, **kwargs):
        hv.extension('bokeh')
        hv.Store.add_style_opts(hv.Table, ['sizing_mode', 'row_height'], backend='bokeh')
        hv.Store.add_style_opts(hv.Curve, ['level'])
        hv.output(widget_location='right')
        return super().__new__(cls)

    def __post_init__(self):
        hv.opts.defaults(
            hv.opts.Curve(max_width=600, height=250),
            hv.opts.Area(max_width=600, height=250),
        )
        for name, val in BokehOpts.__annotations__.items():
            def new_default():
                print(name, val)
                return getattr(self,name,val)

    # Static options
    hist_records: hv.Options = default_factory('hist_records', lambda:
        hv.opts.Histogram(
            alpha=0.75, height=150, responsive=True))

    ensemble_curve: hv.Options = default_factory('ensemble_curve', lambda:
        hv.opts.Curve(
            color='#999999', alpha=0.5, level='underlay', line_width=1,
            tools=['hover'], show_legend=False))

    accent_curve: hv.Options = default_factory('accent_curve', lambda:
        hv.opts.Curve(
            color=sns.color_palette()[3], alpha=1, level='glyph', line_width=2.2,
            tools=['hover'], show_legend=False))

    # For mpl: zorder between ensemble & accent, thicker linewidth
    target_hline: hv.Options = default_factory('target_hline', lambda:
        hv.opts.HLine(
            color='k', alpha=1, line_width=1.5, level='underlay'))

    # FIXME: Also define Curve styling ?
    true_η: hv.Options = default_factory('true_η', lambda:
        hv.opts.Area(
            fill_color="#E8E8E8",
            line_width=0)
    )

    # Dynamic options
    def table(self, nrows) -> hv.Options:
        return hv.opts.Table(row_height=self.row_height,
                             height=(nrows+1)*self.row_height)  # sizing_mode='fixed'

    def dyn_accent_curve(self, Δ, Δthresh=2) -> hv.Options:
        """
        Return a style based on the difference Δ between a curve's log L and
        the log L of the best fitting curve.
        Δ = 0       => accent_curve
        Δ > Δthresh => ensemble_curve
        0 < Δ < Δthresh => linear interpolation between accent and ensemble curve
        """
        if Δ <= 0:
            return self.accent_curve
        elif Δ > Δthresh or not np.isfinite(Δ):
            return self.ensemble_curve
        else:
            accent_color = self.accent_curve.kwargs['color']
            ensemble_color = self.ensemble_curve.kwargs['color']
            accent_alpha = self.accent_curve.kwargs['alpha']
            ensemble_alpha = self.ensemble_curve.kwargs['alpha']
            x = Δ/Δthresh
            color = interpolate_color(accent_color, ensemble_color, x)
            α = (1-x)*accent_alpha + x*ensemble_alpha
            return self.accent_curve(color=color, alpha=α)


# ------------------------------------------------
# ## Mapping identifiers to pretty formatted names

class NameMap(dict):
    """
    A dict with fuzzy hierarchical keys. E.g., if the dict is::

        pretty_names = NameMap({'λθ': 'λ^θ', 'model.λθ': 'λ_{{model}}^θ'})

    then the following would return 'λ^θ'::

        pretty_names['λθ']
        pretty_names['randomtext.λθ']
        pretty_names['model.randomtext.λθ']

    while the following would return 'λ_{{model}}^θ'::

        pretty_names['model.λθ']
        pretty_names['randomtext.model.λθ']

    Moreover, a form of graceful degradation is provided in that both
    ``namemap[key]`` and ``namemap.get(key)`` will return ``key`` if no match
    is found.

    “dollared” versions of names can be obtained by using `dollared` instead
    of `get`. For LaTeX names, this should add a dollar sign before and after
    the name, e.g. "$λ^θ$". For Unicode names, the name should be returned
    unchanged.
    The action of the `dollared` method can be configured with the
    `format_prefix` and `format_suffix` attributes.
    """
    def __init__(self, format_prefix="", format_suffix="",
                 group_prefix="", group_suffix="",
                 subscript_prefix="_", subscript_translator=None,
                 **kwargs):
        self.format_prefix = format_prefix
        self.format_suffix = format_suffix
        self.group_prefix = group_prefix
        self.group_suffix = group_suffix
        self.subscript_prefix = subscript_prefix
        self.subscript_translator = subscript_translator
        super().__init__(**kwargs)
    def __getitem__(self, key):
        while key:
            try:
                return super().__getitem__(key)
            except KeyError:
                if '.' in key:
                    key = key.split('.', 1)[1]
                else:
                    return key
        return key
    def get(self, name, default=None):
        if default is None:
            default = name
        return super().get(name, default)
    def wrap(self, s):
        "Wrap an arbitrary string with the formating prefix & suffix"
        return f"{self.format_prefix}{s}{self.format_suffix}"
    def dollared(self, name, default=None):
        "Return the pretty name wrapped with the formatting prefix & suffix."
        return self.wrap(self.get(name, default))

    def index_str(self, index: Union[str,Tuple[int]], prefix: Optional[str]=None):
        """
        Return a string which can be part of a label to denote the index.
        e.g. `index_str((0, 1))` may return '{{01}}', usable in a LaTeX string.

        A comma is inserted between index values if any oen of them is ≥ 10.

        `index` may be either a tuple or a string.
        `prefix` is added if and only if the index string is non empty.
            It can be used e.g. to add '_' to indicate a suffix.
            If unspecified, self.subscript_prefix is used.
        """
        if prefix is None:
            prefix = self.subscript_prefix
        if isinstance(index, (list, tuple)):
            idx_els = [str(i).strip() for i in index]
        else:
            idx_els = [i.strip() for i in str(index).strip('(),').split(',')]
        if any(len(i) > 1 for i in idx_els):
            sep = ","
        else:
            sep = ""
        idx_str = sep.join(idx_els)
        if self.subscript_translator:
            idx_str = idx_str.translate(self.subscript_translator)
        if prefix and idx_str:
            idx_str = prefix + idx_str
        return idx_str

@dataclass
class NameMapCollection:
    """
    A wrapper for multiple NameMaps, with configurable default so that one
    of them can be accessed directly. Thus the default can be changed once at
    the top of a notebook and all subsequent calls will use the correct NameMap.
    This is motivated by bokeh's lack of support for LaTeX labels.
    """
    latex  : NameMap = field(default_factory=
        lambda: NameMap(format_prefix="$", format_suffix="$",
                        group_prefix="{{", group_suffix="}}"))
    unicode: NameMap = field(default_factory=
        lambda: NameMap(subscript_prefix="",
                        subscript_translator=str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")))
    ascii  : NameMap = field(default_factory=lambda: NameMap())
        # Ascii => no formatting at all, always return the queried name
    default: NameMap = None  # Defaults to 'unicode'; see __post_init__

    def __post_init__(self):
        if self.default is None:
            self.default = self.unicode

    def __getitem__(self, key):
        return self.default.__getitem__(key)
    def get(self, name, default=None):
        return self.default.get(name, default)
    def keys(self):
        return self.default.keys()
    def values(self):
        return self.default.values()
    def wrap(self, s):
        return self.default.wrap(s)
    def dollared(self, key):
        return self.default.dollared(key)
    def index_str(self, index: Union[str,Tuple[int]], prefix: str=""):
        return self.default.index_str(index, prefix)

    def set_default(self, name_map: Union[str,NameMap]):
        "Set the default name map"
        if isinstance(name_map, str):
            name_map = getattr(self, name_map)
        self.default = name_map


# These are used to convert variable identifiers used in code into more nicely
# formatted output
pretty_names = NameMapCollection()
pretty_names.latex.update(
    dict(Wtilde = '\\tilde{{W}}',
         μtilde = '\\tilde{{μ}}',
         τtilde = '\\tilde{{τ}}',
         logτtilde = '\\log\\tilde{{τ}}',
         σtilde = '\\tilde{{σ}}',
         Itilde = '\\tilde{{I}}',
         I      = 'I',
         λθ     = '\\lambda^\\theta',
         λη     = '\\lambda^\\eta',

         fit_hyperparams = '\\Lambda')
    )
pretty_names.unicode.update(
    dict(Wtilde = 'W̃',
         μtilde = 'μ̃',
         τtilde = 'τ̃',
         logτtilde = 'log τ̃',
         σtilde = 'σ̃',
         Itilde = 'Ĩ',
         I      = 'I',
         λθ     = 'λθ',
         λη     = 'λη',

         fit_hyperparams = 'Λ')
    )
param_scales = defaultdict(lambda: 'linear',
                           τtilde  = 'log'
                           )
