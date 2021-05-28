# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
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

# %% [markdown] tags=["remove-cell"]
# > **Structural note**
# > Even though there is only one *composite_models* module, placing it in a directory allows it to be found by [*_scandir.py*](../_scandir.py).

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
import sinnfull
if __name__ == "__main__":
    sinnfull.setup('theano')

# %% tags=["hide-cell"]
from typing import Any, Optional, Union
import numpy as np
import theano_shim as shim
from mackelab_toolbox.typing import FloatX, Shared, Array
from sinn.models import initializer, ModelParams
from sinn.histories import TimeAxis, Series, AutoHist

from sinnfull.models.base import Model, Param
from sinnfull.models.GWN.GWN import GaussianWhiteNoise  # -> NoiseSource
from sinnfull.utils import add_to, add_property_to

# %%
__all__ = ['ObservedDynamics']


# %% [markdown]
# ## Observed dynamics
#
# A model with two submodels – stochastic input and deterministic dynamics – where the state variables of the dynamical model are observed directly.
#
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBpbnB1dHt7aW5wdXQ6IE5vaXNlU291cmNlfX0gLS0-IGR5bmFtaWNzXG4gICAgZHluYW1pY3NbZHluYW1pY3M6IE1vZGVsXSAtLT4gb2JzZXJ2YXRpb25zXG5cbiAgICBzdHlsZSBvYnNlcnZhdGlvbnMgZmlsbDp0cmFuc3BhcmVudCwgc3Ryb2tlLXdpZHRoOjAiLCJtZXJtYWlkIjp7fSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBpbnB1dHt7aW5wdXQ6IE5vaXNlU291cmNlfX0gLS0-IGR5bmFtaWNzXG4gICAgZHluYW1pY3NbZHluYW1pY3M6IE1vZGVsXSAtLT4gb2JzZXJ2YXRpb25zXG5cbiAgICBzdHlsZSBvYnNlcnZhdGlvbnMgZmlsbDp0cmFuc3BhcmVudCwgc3Ryb2tlLXdpZHRoOjAiLCJtZXJtYWlkIjp7fSwidXBkYXRlRWRpdG9yIjpmYWxzZX0)

# %% tags=["remove-cell"]
class SubmodelParams(ModelParams):
    class Config:
        extra = 'allow'
    # Ensure values are always returned in a consistent order (c.f. OptimParams)
    def __iter__(self):
        values = {k: v for k,v in super().__iter__()}
        for k in sorted(values):
            yield k, values[k]
    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        return {k: d[k] for k in sorted(d)}


# %%
class ObservedDynamics(Model):
    time    : TimeAxis
    input   : GaussianWhiteNoise  # -> NoiseSource, or Model
    dynamics: Model
        
    class Parameters(ModelParams):
        input: SubmodelParams
        dynamics: SubmodelParams
        
    def initialize(self, initializer: Union[None,str,tuple,dict]=None):
        if not isinstance(initializer, dict):
            initializer = {'input': initializer, 'dynamics': initializer}
        self.input.initialize(initializer['input'])
        self.dynamics.initialize(initializer['dynamics'])
    
    @classmethod            # argument names should match the submodel names
    def get_test_parameters(cls, input, dynamics, rng=None):
        return cls.Parameters(input=input.get_test_parameters(rng),
                              dynamics=dynamics.get_test_parameters(rng))


# %% [markdown]
# ## Composite priors
#
# Composite priors can be create with PyMC3's mechanism for combining models. The only requirement is to assign to each subprior the name of the corresponding submodel.
#
# For example, the [`ObservedDynamics`](#observed-dynamics) model defines the submodels `input` and `dynamics`; a composite prior for this model might therefore look like:

# %% tags=["hide-cell"]
if __name__ == "__main__":
    import pymc3 as pm
    from sinnfull.models import Prior, priors

    # %% tags=["hide-output"]
    with Prior() as obs_dyn_prior:
        priors.GWN.default(M=2, name="input")
        priors.WC.default(M=2, name="dynamics")

    obs_dyn_prior

# %% [markdown]
# :::{note}  
# Composing subpriors like this works if they are independent. If there are subpriors which depend on others (e.g. because they share a random variable), at present they need to be defined by explicitely listing their random variables, and using `Deterministic` to define dependencies.  
# :::

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

    # Parameters
    rng_sim = get_shim_rng((1,0), exists_ok=True)
        # exists_ok=True allows re-running the cell
    Θ_gwn = paramsets.GWN.default
    Θ_wc  = paramsets.WC.rich
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
        dynamics=dyn,
        params  ={'input': Θ_gwn, 'dynamics': Θ_wc}
    )

# %% [markdown]
# :::{margin}  Formatted representation
# Model representations are nicely formatted in a Jupyter notebook.
# :::

    # %%
    model

    # %%
    # Assert that the noise history (ξ) only gets exported once, and does so
    # tied to the input submodel
    assert 'ξ' in model.dict()['input'].keys()
    assert 'I' not in model.dict()['dynamics'].keys()

    # %%
    # Instead, on export, 'I' is added to the list of connected histories
    assert "dynamics.I" in model.dict()['history_connections'].values()

    # %%
    # Assert that deserialization correctly reconnects histories
    from sinnfull.models import models
    model = models.ObservedDynamics(
        time    =time,
        input   =noise,
        dynamics=dyn,
        params  ={'input': Θ_gwn, 'dynamics': Θ_wc}
    )

    model2 = models.ObservedDynamics.parse_obj(model.dict())
    assert model2.input.ξ is model2.dynamics.I

    model2 = models.ObservedDynamics.parse_raw(model.json())
    assert model2.input.ξ is model2.dynamics.I

    # %% tags=["remove-cell"]
    from IPython.display import display
    import holoviews as hv
    hv.extension('bokeh')

    # %% tags=["hide-output"]
    display(model)

    # %%
    # Initialize & integrate
    model.dynamics.u[-1] = 0
    model.integrate('end')

    # %% tags=["hide-input"]
    # Plot histories
    traces = []
    for hist in model.history_set:
        traces.extend( [hv.Curve(trace, kdims=['time'],
                                 vdims=[f'{hist.name}{i}'])
                        for i, trace in enumerate(hist.traces)] )

    display(hv.Layout(traces).cols(Θ_wc.M))

# %% [markdown]
# The same result can be achieved using the `CreateModel` Task, which provides the `submodels`, `subparams` and `connect` arguments for this purpose. The advantage of using a Task is that nothing is evaluated before `run` is called, which allows the model to be used in a [workflow](/sinnfull/workflows/index).
#
# We also show how to use the composite prior defined above to sample model parameters.

    # %%
    from sinnfull.tasks import CreateModel
    from sinnfull.parameters import ParameterSet
    import smttask; smttask.config.record = False  # Turn off recording for tests

    # %%
    Θ = ParameterSet(obs_dyn_prior.random((3,1)))
        # ParameterSet builds a dictionary by splitting on '.' in RV names
    
    create_model = CreateModel(
        time          =time,
        model_selector= {'__root__'   : {'ObservedDynamics'},
                         'input'      : {'GaussianWhiteNoise'},
                         'dynamics'   : {'WilsonCowan'},
                         '__connect__': ['GaussianWhiteNoise.ξ -> WilsonCowan.I']},
        params       = {'input': Θ.input, 'dynamics': Θ.dynamics},
        rng_key      =(0,1)
    )
    model2 = create_model.run()

    # %%
    # Confirm that the models were connected as specified
    assert model2.input.ξ is model2.dynamics.I

# %%
