# Tasks

*Tasks* are simply function with the following properties:

- They are [_stateless_](https://github.com/alcrene/smttask/blob/master/docs/basics.rst#specifying-a-task), meaning that given the same input, they always return the same output.
- All their input types are specified using [type hints](https://docs.python.org/3/library/typing.html).
- Their output type(s) are also specified using type hints.

In addition, the function must be decorated with one of the Task decorators [provided by _smttask_](https://github.com/alcrene/smttask/blob/master/docs/basics.rst#specifying-a-task).

Tasks in this directory broadly fall into two categories:

Tasks which simply bind parameters
: These are used to avoid storing the output of a function in a task description. For example, the `~sinnfull.tasks.base.CreateSyntheticDataset` Task takes a succinct set of random initialization keys for the model parameter and simulator and returns a complete data set. Since there is often little point in recording these tasks, they are generally `~smttask.MemoizedTask`. They are used as inputs for recorded tasks.

Tasks which do actual work
: These tasks may take substantial computational time, and so are generally `~smttask.RecordedTask`s. Their inputs and outputs are recorded to disk every time they are run; on subsequent runs with the same input, the execution is skipped and the output retrieved from disk.

The proposed structure is to place data access, model creation and optimization tasks within the [_base_](./base.py) module, and analysis tasks in the [_analysis_](./analysis.py). For convenience, all tasks are imported into [*\_\_init\_\_.py*](./__init__.py) – this allows them to be imported elsewhere in the project as `from sinnfull.tasks import <taskname>`.

Creating a new task is quite straightforward, but it may still be helpful to start from one of those included in [_here.py_](./base.py).

## Combining tasks

Tasks can be used as inputs to other tasks: this makes it easy to break down complex analysis sequences into individual tasks, with optional caching of intermediate results if the corresponding tasks are recorded. We call a sequence of tasks a _workflow_. Scripts which construct workflows from individual tasks are placed in the [_workflows_ directory](../workflows).

Since there are no special language or convention for combining tasks, and everything is done in Python, one can use arbitrary code to construct workflows. For example, the [_Optimize_WF_template_](./workflows/Optimize_WF_template) uses a complex validation function to ensure that a model's _stationary_stats_ method is compatible with assumptions of the workflow. Since this test is run when the workflow is created, rather than when it is run, errors are caught much sooner.
