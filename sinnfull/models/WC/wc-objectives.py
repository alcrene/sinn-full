# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # WC objective functions

# %%
import numpy as np
import theano_shim as shim

from sinnfull.models.base import ObjectiveFunction, tag
from sinnfull.typing_ import IndexableNamespace


# %% [markdown]
# Objective functions must follow the usual limitations for being serializable: only use modules defined in `mackelab_toolbox.serialize.config.default_namespace` (these are listed in the cell below), or import additional modules within the function.
#

# %%
# Modules available in global scope when deserializing
import numpy as np
import math
import theano_shim as shim


# %% [markdown]
# ## Public interface
# All functions defined in this module are accessible from the collection `sinnfull.models.objectives.WC`.

# %% [markdown]
# ## Ad-hoc squared-error loss

# %%
@tag.WilsonCowan
@ObjectiveFunction(tags={'se', 'forward'})
def WC_se(self, k):
    "Squared error loss"
    u_predict = self.u_upd(self, k)
    return -((u_predict - self.u(k))**2).sum()
