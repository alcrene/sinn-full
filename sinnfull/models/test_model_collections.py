# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # Filtering model collections with tags
#
# Tests / usage examples for model collections.
#
# :::{margin}  
# See also [](model-organization), [](tags-taming-model-proliferation).  
# :::
#
# `sinnfull.models` defines four global collections:
#
# - `sinnfull.models.models`
# - `sinnfull.models.paramsets`
# - `sinnfull.models.priors`
# - `sinnfull.models.objectives`
#
# These are populated automatically by scanning subdirectories, and can be filtered by tags.

# %% tags=["remove-cell"]
import sinnfull
sinnfull.setup('numpy')

# %%
from sinnfull.models import models, priors, paramsets, objectives

# %%
print(models.tags)
print(priors.tags)
print(paramsets.tags)
print(objectives.tags)

# %%
models

# %%
models.OU

# %% [markdown]
# When only one element remains after filtering, the list is automatically squeezed.
# This avoids the need to do unsightly things like `models.WC[0]`.  

# %%
models.WC

# %% [markdown]
# Nevertheless, overspecifying tags is not a problem.

# %%
models.WC.WilsonCowan

# %% [markdown]
# Filtering can also be done with key based indexing.

# %%
objectives['forward']

# %% [markdown]
# And of course the two styles can be mixed.

# %%
objectives['forward'].OU_AR

# %% [markdown]
# Key-based filtering allows specifying multiple tags at once.

# %%
objectives['forward', 'OU_AR']     # Both are equivalent
objectives[{'forward', 'OU_AR'}]   # to the one above

# %% [markdown]
# One can always fall back to indexing the underlying list with integers.
# However the ordering of this list is fragile since it depends on module import order.

# %%
objectives['forward'].OU_AR[0]

# %% [markdown]
# Tags can be negated by prepending "!", which is especially useful when one set of tags is a subset of another.

# %%
objectives['forward'].OU_AR["!backward"]

# %%
paramsets.WC

# %%
priors.default

# %%
priors.dale

# %%
