# Models

Models are composed of three types of objects:

- A _model_
  - Inherits from `~sinn.Model`.
  - Defines the dynamical equations.
  - Its parameters are defined by _model.params_ (which inherits from `~sinn.ModelParams`).
- A _prior_
  - Inherits from `~sinnfull.models.base.Prior`.
  - Sets constraints on the allowable parameters.
  - Used to define transformations from the model parameters to others which are more suited for optimization.
  - Used to set which parameters to optimize, by making the prior on fixed parameters a `~pymc3.Constant`.
  - Can also be used to define a regularizer, since mathematically prior and regularize are equivalent.
- An _objective_
  - Inherits from `~sinnfull.models.base.ObjectiveFunction`.
  - Combined with the _prior_, defines the objective that will be optimized.
  - Equated to the log probability of the parameters given the observations.  
    Accordingly, the combination with the prior is simply `objective + prior.logp`.
  - Should be _maximized_ to obtain optimal parameters.

(model-organization)=
## Organization

Since priors and objectives defined for one model are unlikely to work for another, they are defined alongside models in subdirectories (e.g. [Ornstein-Uhlenbeck](./OU/OU), [Ricker](./Ricker/Ricker), [Wilson-Cowan](./WC/WC)). The base classes are defined in [base.py](./base.py).

`sinnfull.models` defines four global collections:

- `sinnfull.models.models`
- `sinnfull.models.paramsets`
- `sinnfull.models.priors`
- `sinnfull.models.objectives`

These are populated automatically by scanning subdirectories, and can be filtered by tags (see [below](#taming-model-proliferation-with-tags)), which makes it easier to discover which definitions are available. Objects in these collections always have at least one tag, namely the name of the model they are associated with (inferred from the directory name).

For illustrative purposes, the following is a hypothetical folder sctructure with two types of models, “OU” and “other_model”, and at least three named parameter sets for OU model: “defaults”, “symmetric” and “low-noise”.

    sinnfull
    │ ...
    └─ models
      ├─ OU
      │  ├─ defaults.paramset
      │  ├─ symmetric.paramset
      │  ├─ low-noise.paramset
      │  ├─ model.py
      │  └─ objectives.py
      └─ other_model
         │ ...

:::{note}  
In order to populate the collections, everything within each model subpackage is imported. For this framework's target application of focused projects with small sets of models, this is convenient, but could become a problem if the number of models grows large.
:::

(tags-taming-model-proliferation)=
## Taming model proliferation with tags

One issue which occurs is that of proliferation: for each model, there can be multiple parameter sets, priors and objectives. Defining a new model for each combination quickly becomes a managerial nightmare. Separating models into modular components, namely the _model_, _prior_ and _objective_ classes described above, is one way to reduce the combinatorics of this task.

:::{margin}
A third technique used to reduce proliferation is the support for [arithmetic on objective functions](objective-functions).
:::

Another way, which is especially useful to access definitions in a discoverable way, is to use _tags_. These are arbitrary strings associated with an object. Objects can then be self-assembled into [`TaggedCollection`](/sinnfull/tagging), which can be filtered by any tag combination. For example, if we have two Wilson-Cowan models named `WC1` and `WC2` with equivalent parameters, we might tag their default parameter set with `"WC1"`, `"WC2"` and `"default"`. A different parameter sets, meant to reproduce high gamma oscillations, might be tagged `"WC1"`, `"WC2"` and `"highgamma"`. The first set could then be retrieved with

```python
from sinnfull.models.WC import paramsets
params = paramsets.WC1.default
# params = paramsets.default.WC1  # Equivalent
```

To list all parameter sets tagged `"highgamma"`, one would do

```python
paramsets.highgamma
```

And the following would return a dictionary with all parameter sets, organized by tags:

```python
paramsets.by_tag()
```

See also the [example notebook](./test_model_collections).

### Automatically assigned tags

A few tags are automatically assigned:

- The name of a subdirectory is assigned to all objects defined within it.
- The filename of a `ParameterSet` is assigned to it (if defined in a *.paramset* file)
- The variable name of a `ParameterSet` is assigned to it (if defined in a module)
