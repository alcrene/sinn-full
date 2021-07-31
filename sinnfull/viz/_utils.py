# -*- coding: utf-8 -*-
# # Visualization utilities

import logging
from typing import Optional, List, Tuple
import numpy as np
import matplotlib as mpl
import holoviews as hv

logger = logging.getLogger(__name__)

# ## Interpolating colors
#
# Interpolate between two colors in HSV color space.

def interpolate_color(c0, c1, x: float):
    """
    Interpolate between two colors in HSV color space.
    This works best if c0 and c1 have similar hue, so that interpolation is
    done on saturation and value.
    :param:c0: Color for x = 0
    :param:c1: Color for x = 1
    :param:x: Value between 0 and 1
    """
    assert 0 <= x <= 1
    hsv0 = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(c0))
    hsv1 = mpl.colors.rgb_to_hsv(mpl.colors.to_rgb(c1))
    # If either one of the colors is a shade of grey, its hue is meaningless
    # => set it equal to the other's hue so interpolation doesn't change color
    if hsv0[1] == 0:
        hsv0[0] = hsv1[0]
    if hsv1[1] == 0:
        hsv1[0] = hsv0[0]
    # The hue is periodic: Going from 0 to .9 should be replaced by 0 to -.1
    # The % 1 after interpolation brings us back to the [0,1] range
    if hsv1[0] - hsv0[0] > 0.5:
        hsv1[0] -= 1
    if hsv0[0] - hsv1[0] > 0.5:
        hsv0[0] -= 1
    # Interpolate and convert to hex string
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb(((1-x)*hsv0 + x*hsv1) % 1))

# ## Get $\log L$ quantile
#
# Return the value of the requested quantile for the values of `data`.
#
# This is used to find fits which have high likelihood overall, rather than just at the time the fit terminated.

def get_logL_quantile(data: List[Tuple[int,float]], quantile: float=.95,
                      window: Optional[int]=None
                     ) -> float:
    """
    Return the value of the requested quantile for the y values of `data`.

    Parameters
    ----------
    data: List of (step, log L) pairs.
        Given an `hv.Curve` object for likelihood evolution, this is its
        `.data` attribute.
    quantile: The logL value at this quantile assigned as a fit's 'log L'
        value (see above).
    window: Number of steps from the end to consider.
        By default, all steps are considered when determining the log L value;
        early steps can be discarded by setting this to a positive value. For
        example, if `window`=1000, only log L values in the **last** 1000 steps
        are considered.
        This can help discard initial fluctuations to focus on stabilized
        log L values, which are usually more relevant.

    Returns
    -------
    float: The quantile value.
        If any of the data points are non-finite, returns -∞.
    """
    # Argument validation
    if quantile < 0.7 and window is None:
        logger.warning("When using a small value of `quantile` it is "
                       "recommended to also set a value for `window` in "
                       "order to discard the initial transients.")
    if window is None:
        window = np.inf
    # We need to make data points regularly spaced in order to perform statistics
    # on the y values. We create a new array of 4096 regularly spaced steps
    # (4096=2**12 is arbitrary but since this is cheap, we might as well take a
    # large number and avoid rounding errors.)
    data = np.array(data)
    if not np.isfinite(data).all():
        return -np.inf
    x0, xn = data[0,0], data[-1,0]
    x = np.linspace(max(x0, xn-window), xn, 4096)  # Interpolate the curve at these points
    y = np.interp(x, data[:,0], data[:,1])
    return np.quantile(y, quantile)

# ## Converting `DynamicMap` layout to `HoloMap` layout
#
# HoloViews _should_ be able to convert a `Layout` of `DynamicMap` elements to a `Layout` of `HoloMap` elements automatically, but we haven't found this to be reliable. This function essentially does the job manually: by iterating over the Layout panels, constructing HoloMaps for each, and assembling them into a new Layout.
#
# `kwargs` are used to reduce the number of frames computed for static HoloMap.

from typing import Collection, Optional, Dict
from tqdm.auto import tqdm

def convert_dynmap_layout_to_holomap_layout(
    dynmap_layout: hv.Layout,
    all_frame_keys: Collection[tuple],
    include: Optional[Dict[str,Collection]]=None,
    exclude: Optional[Dict[str,Collection]]=None) -> hv.Layout:
    """
    HoloViews _should_ be able to convert a `Layout` of `DynamicMap` elements to
    a `Layout` of `HoloMap` elements automatically, but we haven't found this to
    be reliable. This function essentially does the job manually: by iterating
    over the Layout panels, constructing HoloMaps for each, and assembling them
    into a new Layout.

    `kwargs` are used to reduce the number of frames computed for static HoloMap.

    .. Todo:: Currently, if some dimensions have default values, those values
       must be included in the holomap (otherwise KeyError is raised).

    Parameters
    ----------
    dynmap_layout: Must be a `hv.Layout` where each panel is a `hv.DynamicMap`.
        Each `DynamicMap` should have the same keys.
    all_frame_keys: Complete list of possible keys.
        Because DynamicMaps don't provide a full list of keys, this must be
        provided separately.
        Note that even if key dimensions all provide values, it is not always
        correct to list their outer product, as some combinations may be invalid.
        TODO: Provide outer product as default ?
    include: Which values to include. Keys must match the dimension names
        and values should be collections of values for that dimension. E.g.::
            (..., include={'step': 0, 5, 1000})
    exclude: Values to exclude, even if they match `include`.
        Default value is `{'latents': ['∅']}`, which excludes fits with no latents.
    """
    assert all(isinstance(dynmap, hv.DynamicMap) for dynmap in dynmap_layout)
    if include is not None:
        assert isinstance(include, dict)
        assert all(isinstance(p, Collection) for p in include.values())
    if exclude is None:
        exclude = {'latents': ['∅']}
    else:
        assert isinstance(exclude, dict)
        assert all(isinstance(p, Collection) for p in exclude.values())
    # Get the list of key dimensions; they must be the same for each dynamic map panel
    dynmaps = {panelkey: dynmap for panelkey, dynmap in dynmap_layout.items()}
    kdims = next(iter(dynmaps.values())).kdims
    kdim_names = [dim.name for dim in kdims]
    assert all(kdims == dynmap.kdims for dynmap in dynmaps.values())

    # Create the frame selector by first listing all possible keys, and then filtering
    # with the conditions given by include & exclude
    # NOTE: We avoid using an outer product because some combinations may be impossible
    # We convert filters to sets b/c those have faster inclusion tests compared to lists
    # FIXME: Warn if not all panels have the same frame keys ?
    frame_selector = set(all_frame_keys)
    if include is not None:
        frame_include_filter = {k: set(v) for k, v in include.items() if k in kdim_names}

        if set(include) - set(frame_include_filter):
            logger.warning("The following keys in `include` don't match any dimension name:\n"
                 f"{set(include) - set(frame_include_filter)}")
    else:
        frame_include_filter = {}
    frame_exclude_filter = {k: set(v) for k, v in exclude.items() if k in kdim_names}
        # Sets have faster inclusion tests compared to lists
    if set(exclude) - set(frame_exclude_filter):
        logger.warning("The following keys in `exclude` don't match any dimension name:\n"
                       f"{set(exclude) - set(frame_exclude_filter)}")
    if frame_include_filter:
        for kdim_name, include_values in frame_include_filter.items():
            i = kdim_names.index(kdim_name)
            for frame_key in frame_selector.copy():
                if frame_key[i] not in include_values:
                    frame_selector.remove(frame_key)
    if frame_exclude_filter:
        for kdim_name, exclude_values in frame_exclude_filter.items():
            i = kdim_names.index(kdim_name)
            for frame_key in frame_selector.copy():
                if frame_key[i] in exclude_values:
                    frame_selector.remove(frame_key)

    data = {panelkey: {} for panelkey in dynmap_layout.keys()}
    values = {name: set() for name in kdim_names}
    if len(frame_selector) == 0:
        logger.error("Combination of Include & exclude conditions lead to "
                     "all frames being excluded. No Holomap is returned.\n"
                     f"Include: {include}\nExclude: {exclude}")
        return
    for sel in tqdm(sorted(frame_selector), desc="Computing frames"):
        # Putting the panel loop inside means that caches shared across panels are reused
        # (which happens when panels are different variable components of the same model)
        for panelkey, panel in dynmap_layout.items():
            data[panelkey][sel] = panel[sel]
        for dim, value in zip(kdims, sel):
            values[dim.name].add(value)
    holomap_layout = dynmap_layout.clone(
        [hv.HoloMap(panel_data, kdims=panel.kdims).redim.values(**values)
         for panel_data in data.values()])

    return holomap_layout
