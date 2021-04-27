# Diagnostic toolkit

The scripts and notebooks in this folder collect functions which at some point were found to be useful in tracking down bugs or unintuitive behaviour. They may have related to issues with the general library code, or with the user level code (e.g. a bug in the user-defined log likelihood function).

In most cases the code is not “ready-to-use”, but rather an example that should be adapted as needed. Since they include hard-coded values, it may be desirable to use [`git update-index --skip-worktree`](https://compiledsuccessfully.dev/git-skip-worktree/) to tell git to ignore changes in them. Diagnostic utilities that don't contain hard-coded values, and therefore _should_ be version-controlled, are located under [sinnfull/diagnostics](../diagnostics/__init__).

Scripts including _“— example”_ in their names are similar to small tutorials: they are tied to specific example fits, which allows them to more concretely illustrate how to interpret tests. These are essentially archived analyses, and are typically _not_ excluded from git's change tracking. When using them, you may want to work on a copy to keep the original.

Included scripts address the following issues. Those in **bold** have the highest likelihood of being useful. The others may be out of date.

- **Checking correctness of implementations**
  - **[Check that the optimized likelihood matches the simulator's p.d.f.](./One-step-forward%20cost)**
  - **[Check correctness of simulators by comparing to stationary stats](./Check%20stationary%20stats)**
  - [Inspect gradients for the latent variable at different points of the fit](./Inspect_gradients)
  - [Inspect and plot marginals of the log likelihood](./Inspect_logL)
- **Testing hyperparameters**
  - **[Check that fits without latent variables converge](./Fit_stability.ipynb)**
  - [Investigate the effect of hyperparameters](./Inspect_hyperparams)
- **Understanding fit dynamics**
  - **[Check uniformity of gradients – example](./Check%20uniformity%20of%20gradients%20--%20example.ipynb)**
- **Finding why runs differ**
  - **[Find which parameters changed between two runs](./Compare_parameters.ipynb)**
  - **[Find the difference between two task files](./compare_tasks.py)** \
    Typical use-case: determining which parameters is preventing a task from serializing consistently
- **Debugging**
  - **[Debugging runtime warnings](./Debugging_runtime_warnings.md)** \
    More advanced tricks that may be useful for tracking down e.g. numerical instabilities due to a model's parameterization.
