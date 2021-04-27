# Optimizers

These subpackage contains four types of objects:

- Optimizers
- Recorders
- Convergence tests
- Optimization parameters

Optimizers
~ implement an optimization algorithm, and are by far the most voluminous classes. General documentation and the abstract base class `Optimizer` are given in [*base.py*](./base); concrete implementations are found under [*optimizers*](./optimizers/index).

  Optimizers all provide a `step()` method, which iterates the optimization algorithm one step forward.

Recorders
~ are executed after optimization steps to record various quantities (e.g. loss or parameter values). They provide both `ready` and `record` methods; the former allows each recorder to set its own recording frequency.

  See [*Recorders*](./recorders) for details.

Convergence tests
~ are used to add early stopping conditions. Like _recorders_, they are executed after optimization steps.

  Available tests are defined in [*convergence_tests.py*](./convergence_tests).

Optimization parameters
~ set hyperparameters like learning rate and batch size. Since these tend to be model-specific, they are organized into separate directories – directory names should match the model they are intended for. All of these directories are under [*paramsets*](./paramsets).

  Like [model parameters](tags-taming-model-proliferation), these are organized into `TaggedCollections`. Collections can be filtered based on model name, and additional tags specified in the same way as for model parameters.
