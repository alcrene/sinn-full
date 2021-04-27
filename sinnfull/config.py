# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

# ---- Constants ------ #

from mackelab_toolbox.utils import sentinel

_NoValue = sentinel("<no value - sinnfull>")

# ------ Flags -------- #

diagnostic_hooks = None
view_only = False   # Set by sinnfull.__init__.setup()
                    # A True value indicates that no tasks will be executed,
                    # so e.g. Recorders don't need to bother binding themselves
                    # to the optimizer.
