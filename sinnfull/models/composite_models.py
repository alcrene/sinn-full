# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
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
# # Composite models
#
# It is often the case that one desires to use the output of one model as the input to another. However, defining a separate model for each such combination quickly becomes cumbersome. _Composite models_ are intended to avoid this need, and thereby help reduce model proliferation. They are composed of two or more _submodels_.
#
# :::{note}  
# Dependencies between submodels must form an directed acyclic graph: if one of the histories of model A is used in model B, than model A must not depend on _any_ history in model B, even if the mathematical equations would allow it.
#
# This assumption greatly simplies building the composite model, and allows conditional log probabilities to trivially factorize across models.   
# :::

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('numpy')

# %% tags=["hide-cell"]
from typing import Any, Optional, Union
import numpy as np
import theano_shim as shim
from mackelab_toolbox.typing import FloatX, Shared, Array
from sinn.models import initializer
from sinn.histories import TimeAxis, Series, AutoHist

from sinnfull.models.base import Model, Param
from sinnfull.models.GWN.GWN import GaussianWhiteNoise  # -> NoiseSource

# %%
__all__ = ['ObservedDynamics']


# %% [markdown]
# ## Observed dynamics
#
# A model with two submodels – stochastic input and deterministic dynamics – where the state variables of the dynamical model are observed directly.
#
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBpbnB1dHt7aW5wdXQ6IE5vaXNlU291cmNlfX0gLS0-IGR5bmFtaWNzXG4gICAgZHluYW1pY3NbZHluYW1pY3M6IE1vZGVsXSAtLT4gb2JzZXJ2YXRpb25zXG5cbiAgICBzdHlsZSBvYnNlcnZhdGlvbnMgZmlsbDp0cmFuc3BhcmVudCwgc3Ryb2tlLXdpZHRoOjAiLCJtZXJtYWlkIjp7fSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBpbnB1dHt7aW5wdXQ6IE5vaXNlU291cmNlfX0gLS0-IGR5bmFtaWNzXG4gICAgZHluYW1pY3NbZHluYW1pY3M6IE1vZGVsXSAtLT4gb2JzZXJ2YXRpb25zXG5cbiAgICBzdHlsZSBvYnNlcnZhdGlvbnMgZmlsbDp0cmFuc3BhcmVudCwgc3Ryb2tlLXdpZHRoOjAiLCJtZXJtYWlkIjp7fSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)

# %%
class ObservedDynamics(Model):
    time    : TimeAxis
    input   : GaussianWhiteNoise  # -> NoiseSource, or Model
    dynamics: Model
        
    def initialize(self, initializer=None):
        self.input.initialize(initializer)
        self.dynamics.initialize(initializer)


# %% [markdown]
# ## Tests & examples

# %% [markdown]
# In an interactive session, the simplest way to construct a composite model is to create the submodels first, and then assemble them into the composite. In the example below, we create an `ObservedDynamics` model using a `GaussianWhiteNoise` model for the input and a `WilsonCowan` model for the dynamics.
#
# Note how the output `ξ` of `GaussianWhiteNoise` is connected to the input `I` of `WilsonCowan`.

# %%
if __name__ == "__main__":
    from sinnfull.models import paramsets
    from sinnfull.models.GWN.GWN import GaussianWhiteNoise
    from sinnfull.models.WC.WC import WilsonCowan
    from sinnfull.rng import get_shim_rng
    from IPython.display import display
    import holoviews as hv
    hv.extension('bokeh')

    # Parameters
    rng_sim = get_shim_rng((1,0), exists_ok=True)
        # exists_ok=True allows re-running the cell
    Θ_wc  = paramsets.WC.rich
    Θ_gwn = paramsets.GWN.rich
    assert Θ_wc.M == Θ_gwn.M
    time  = TimeAxis(min=0, max=.4, step=2**-10)

    # Model
    noise = GaussianWhiteNoise(
        time  =time,
        params=Θ_gwn,
        rng   =rng_sim
    )
    dyn = WilsonCowan(
        time  =time,
        params=Θ_wc,
        I     =noise.ξ
    )

    model = ObservedDynamics(
        time    =time,
        input   =noise,
        dynamics=dyn
    )

    # %%
    # Initialize & integrate
    model.dynamics.u[-1] = 0
    model.integrate('end')

    # %%
    # Plot histories
    traces = []
    for hist in model.history_set:
        traces.extend( [hv.Curve(trace, kdims=['time'],
                                 vdims=[f'{hist.name}{i}'])
                        for i, trace in enumerate(hist.traces)] )

    display(hv.Layout(traces).cols(Θ_wc.M))

# %% [markdown]
# The same result can be achieved using the `CreateModel` Task, which provides the `submodels`, `subparams` and `connect` arguments for this purpose. The advantage of using a Task is that nothing is evaluated before `run` is called, which allows the model to be used in a [workflow](/sinnfull/workflows/index).

    # %%
    from sinnfull.tasks import CreateModel
    import smttask; smttask.config.record = False  # Turn off recording for tests

    # %%
    create_model = CreateModel(
        time   =time,
        model  ='ObservedDynamics',
        params ={},
        rng_key=(0,1),
        submodels = {'input':'GaussianWhiteNoise',
                     'dynamics': 'WilsonCowan'},
        subparams = {'input': Θ_gwn, 'dynamics': Θ_wc},
        connect=['GaussianWhiteNoise.ξ -> WilsonCowan.I']
    )
    model2 = create_model.run()

    # %%
    # Confirm that the models were connected as specified
    assert model2.input.ξ is model2.dynamics.I
