# # HoloViews plotting hooks
#
# [Plot hooks](https://holoviews.org/user_guide/Customizing_Plots.html#Plot-hooks) are used to interact directly with plot objects, for functionality not exposed by HoloViews.

def hide_table_index(plot, element):
    plot.handles['table'].index_position = None
