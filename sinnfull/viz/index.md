# Visualization utilities

Standard visualization functions for:

- Understanding fit dynamics and reporting their performance.
- Aggregating and displaying run metadata (e.g. runtime, conditions, outcome)
- Visualizing data sets

## Top-level objects

The following objects are accessible as `sinfull.viz.<name>`. To add more (e.g. plotting functions for your data), edit the [*\_\_init\_\_.py*](./__init__.py) module.

[`sinnfull.viz.config`](./config)  
~ Configuration options

[`pretty_names`](./config)
~ A mapping from variable identifier to formatted strings. Used to improve the display of variables in plots.
~ Mappings to both LaTeX and Unicode can be defined. The former is preferred for matplotlib and pandas, while the second is very much preferred for Bokeh.

[`BokehOpts`](./config)
~ Global defaults for Bokeh figures.

The following types are provided for inspecting the record store.

`RSView`
~ Record store viewer.

`FitData`
~ Collect all data required for plotting.
~ Plotting functions for individual elements (curves for single fits).

`FitCollection`
~ Dictionary of `FitData` objects.
~ Plotting functions for ensembles of records (from individual elements).

`ColorEvolCurvesByMaxL`, `ColorEvolCurvesByInitKey`
~ Functions used to highlight particular curves in ensemble plots.

:::{hint}  
The [`StrTuple`](./typing_) type can be used to convert a tuple into a key that behaves well with HoloViews.  
:::
