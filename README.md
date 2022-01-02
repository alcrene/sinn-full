# Sinn-full project template

A complete, end-to-end workflow template for inferring dynamical models, so you can focus on the model rather than the implementation.

## Features

- **Fully-functioning examples**  
  All dependencies are installed automatically, so you can get going immediately.
- **Modular workflows**  
  Add new data sources, models or optimizers components without changing those that already suit your problem.
- **Modular models**  
  Dynamical systems are defined in a way that closely parallels their equations, thanks to the functionality of [Sinn](https://github.com/mackelab/sinn).
  + Only the parameters and update equations need to be specified.
    + In particular, each equation is defined _separately_, which means they can also be tested separetely. This makes extending and modifying complex models _much_ easier.
  + _Sinn_ models automatically assemble their equations into a complete simulator, which can be called with their  `integrate` method.
  + Equations are translated to C-code and compiled using [aesara](https://aesara.readthedocs.io), enabling fast simulation.
    + Translation can be done with either the default Theano generator or the newer (but experimental) [JAX generator](https://docs.pymc.io/notebooks/GLM-hierarchical-jax.html).
  + **Objective functions can be automatically differentiated through the entire simulator,** enabling the optimization of complex model with analytical gradients.
- Fully compatible with [PyMC3](https://pymc.io).
- Built-in, obsessive **reproducibility**.
  + Every parameter, code version and workflow sequence is automatically recorded with [Sumatra](https://sumatra.readthedocs.io/), through [smttask](https://github.com/alcrene/smttask).
  + This includes tracking of various random numbers used to generate models and fit them.
    + Functions are [provided](./sinnfull/rng) to convert succinct tuple into high-entroy, independent random number generators.
- Additional utilities:
  + Various [visualization functions](./sinnfull/viz/index).
  + A [diagnostics toolkit](./sinnfull/diagnose/index) to help track down inevitable bugs.

## Getting started

1. Click the [“Use this template” button](https://github.com/alcrene/sinnvoll/generate) in the GitHub interface.
2. Choose a project name and answer the prompts.
3. Clone your new project onto your computer.
4. From within the local copy, run \
   `python3 rename.py <projectname>` \
   This will replace all occurrences of “sinnfull” with your chosen project name.
5. Run the _install.sh_ script.
   - This will work on Linux and MacOS. If you are on Windows, or prefer to know what you are installing, see the [Installation](#installation) instructions below.
6. Run `smttask init`, as indicated by the installation script.
7. (Optional but recommended) [Change the master entropy value](sinnfull/rng.py).

You are now ready to start running the examples and to develop your own models.

:::{hint}  
The pages of this book were generated directly from Sinn-full's source code, so they can be used as a code browser.  
:::

:::{attention}  
This template is still in beta. If you find an example which doesn't work, please open an [issue](https://github.com/alcrene/sinnvoll/issues/new) !  
:::

:::{note}  
To ease reading, not all code cells are included in these generated pages – in particular, boilerplate imports are generally skipped. If you are trying to reproduce outputs, make sure to start from the actual source (*.py*) files !  
:::

## Relationship to *Sinn*

[*Sinn*](https://github.com/mackelab/sinn) was built to address a specific goal: to construct differentiable objective functions from dynamical models, and thus allow their optimization with the same technology as that used to train neural networks. While it is possible to [use *Sinn* on its own](https://github.com/mackelab/sinn/tree/master/examples/Random-RNN), we found that it was difficult for users to translate this into workflows for their own problems. And with good reason – there is a lot more to a good machine learning workflow than just defining an objective function !

*Sinn-full* is a complete workflow intended to get you (and ourselves !) started with a new *Sinn* project as quickly as possible. The intention is to use *Sinn-full* as a starting point, adding and changing elements as dictated by the requirements of the projects. And since *Sinn-full* includes examples that run out of the box, each change can be tested immediately, leading to faster and more confident development.

## Running examples

### Run a single example

- Navigate to _workflows_ and open the notebook [_Optimize_WF_template.py_](./sinnfull/workflows/Optimize_WF_template.py) (it is saved as a Python module but is in fact a notebook – see [Literate programming with Jupytext](#literate-programming-with-jupytext)).
- Optionally change some of the parameters.
- Click the notebook's “Run all cells” button.
- To later create batches of runs, open the [command palette](https://jupyterlab.readthedocs.io/en/stable/user/commands.html) and ensure that a checkmark is present beside “Pair Notebook with ipynb file”. Now save the file to create the paired notebook _Optimize_WF_template.ipynb_.

::: {note}  
The same file – _Optimize_WF_template_ – is used both for single, in-the-notebook execution, and to create  batches of runs. This allows you to execute and debug your workflow directly in a notebook, and then use that same notebook to generate low-running parameter sweeps – without changing a single line of code.  
:::

::: {hint}
The _Optimize_WF_template_ is the highest-level description of the inference problem, and the file you are mostly likely to modify regularly. It is worth reading through the inlined explanations to understand how model, optimizer and objective function are combined into an optimization task.  
:::

### Run an example workflow

- Ensure you've paired the notebook with an ipynb file, as outlined [above](#to-run-a-single-example).
- Navigate to _workflows_ and open the file [_Generate tasks.py_](./sinnfull/workflows/Generate%20tasks.py) as a notebook.
- Optionally change some parameters, then run the notebook. This executes the code file [_Optimize_WF_template.ipynb_](./sinnfull/workflows/Optimize_WF_template.ipynb) for every parameter set.
  + Each time this file is executed, it creates a [_task file_](https://github.com/alcrene/smttask), which is saved in a directory _tasklist_.
  + Task files are self-contained units of work. They can for example be created on one machine and executed on another.
- From a console, navigate to the directory _workflows/tasklist_ and execute  

      smttask run *
      
  to execute all task files in the directory.
  A variety of run options options are provide, which can be listed with `smttask run --help` to see them. The `-n` option in particular allows to specify the number of employed CPU cores, and the `--pdb` option can be used to debug a script.

::: {note}  
The `smttask run` command is meant as a convenient command for executing small batches of tasks. Although it provides basic multiprocessing capabilities, it is not intended to replace a full blown task scheduler like [snakemake](https://snakemake.readthedocs.io/en/stable/) or [doit](https://pydoit.org/). To use a scheduler, create a job for that scheduler which calls `smttask run` on the desired task file(s).[^scheduler]
:::

[^scheduler]: For more fine-grained control, instead of using the command line interface of `smttask run`, a job can also load tasks directly and call their `.run()` method. (Presuming the scheduler can execute Python code.)

### View fit results

- Execute some inference runs (for example, by [running the example workflow](#to-run-an-example-workflow)).
- Open the [_Result viewer_](./sinnfull/view/Result%20viewer.py) notebook
- Run all code cells.

## Adapting the template to your project

### Define your own data sources

See [data](./sinnfull/data/index).

### See available models

In a Python session, type

```python
import sinnfull.models
sinnfull.models.list()
```

Alternatively, browse the files in the [_models_](./sinnfull/models/index) directory.

### Create your own models

Copy one of the model definition files, e.g. the [Wilson Cowan model](./sinnfull/models/WC/WC.py), and change as required.

### Create new analysis tasks

*Tasks* are simply function with the following properties:

- They are [_stateless_](https://github.com/alcrene/smttask/blob/master/docs/basics.rst#specifying-a-task), meaning that given the same input, they always return the same output.
- All their input types are specified using [type hints](https://docs.python.org/3/library/typing.html).
- Their output type(s) are also specified using type hints.

In addition, to convert a function into a Task, must be decorated with one of the Task decorators [provided by _Smttask_](https://github.com/alcrene/smttask/blob/master/docs/basics.rst#specifying-a-task).

More detailed instructions can be found in the [tasks directory](./sinnfull/tasks/index.md). The [Smttask documentation](https://github.com/alcrene/smttask/tree/master/docs) may also be useful.

To assemble multiple tasks into a workflow, create a new file in [_workflows_](./sinnfull/workflows/index). The [_Optimize_WF_template_](./sinnfull/workflows/Optimize_WF_template) file is a good starting point.

### Create a new optimizer

The included optimizer defined under [optim](./sinnfull/optim/index) has extensive inline documentation explaining its algorithm and parameters. Use it as a template and adapt to your needs.

Ensure your optimizer inherits from `sinnfull.Optimizer` and that the module it is defined in is imported by `sinnfull.optim.__init__.py`. This will ensure that it is added to the `sinnfull.optim.optimizers` dictionary.

After creating a new optimizer, you will need to update your [workflow file](./sinnfull/workflows/Optimize_WF_template.py) to use it.

## Publishing

### Fill in project data

The following files should be updated with your project data:

- *README.md* (this file)
- *setup.py*
- *_config.yml*
- *_toc.yml*

Look especially for chevrons `<<<` and expressions bracketed with `{> … <}`.

(The *_toc.yml* file can also be autogenerated, as described below.)

### Generate the HTML project browser

This project's [HTML pages](https://sinn-full.readthedocs.io) were created with [JupyterBook](https://jupyterbook.org/file-types/jupytext.html?highlight=jupytext). They can be recreated from your local copy by navigating to the directory containing this README and typing
```
jb build .
```

The output is placed in a directory *_build*.

You can configure the output by modifying the *_config.yml* and *_toc.yml* files.

To create a new [table of contents](https://jupyterbook.org/customize/toc.html) (toc) by scanning through the directories, the following commands provides a reasonable starting point:

```
jb toc . && grep -v __init__ _toc.yml | grep -v setup | grep -v "/_" | grep -v config | grep -v utils | grep -v rename > __toc.yml && mv __toc.yml _toc.yml
```

(The first command `jb toc .` creates the table of contents, and the other commands remove various undesired patters.) The resulting *_toc.yml* will likely still require some editing, for example to set the order of sections.

### Publishing to Read the Docs

**Quick start:**

- Install [pre-commit](https://pre-commit.com/).

      pip install --user pre-commit
      pre-commit install
      
- Add your package to [env-docs.yaml](./env-docs.yaml) (look for the `>>>>>` signpost).
- Commit your change and push.
- [Import](https://docs.readthedocs.io/en/stable/intro/import-guide.html) the project using Read the Docs' web interface.

**Explanation:**

The configuration format required by Jupyter Book and Read the Docs is slightly different; [pre-commit](https://pre-commit.com/) is used to add a git hook: every time a new commit is made, it translates the Jupyter Book configuration at *_config.yml* and creates/updates the file *conf.py*. If this file changes, the commit is blocked: you need to also commit *conf.py*, thus ensuring that *_config.yml* and *conf.py* always stay in sync.
For more information, see the relevant page in the [Jupyter Book docs](https://jupyterbook.org/publish/readthedocs.html); the blurb in the [RTD docs](https://docs.readthedocs.io/en/stable/faq.html#how-can-i-deploy-jupyter-book-projects-on-read-the-docs) and the Jupyter Book's documentation on [Sphinx usage](https://jupyterbook.org/sphinx/index.html#sphinx-usage-and-customization) may also be useful.

This project template already contains default files *.readthedocs.yaml* and *.pre-commit-config.yaml*, required to configure *Read the Docs* and *pre-commit* respectively – these should work in most cases, but you should check that they correspond to your needs.

:::{note}
Free accounts on Read the Docs only support public repositories.
:::

### Other options for hosted publishing

It is also possible to generate the HTML yourself and host it on your own server or on a service like [GitHub Pages](https://docs.github.com/en/github/working-with-github-pages). In the latter case, it is also possible to setup a GitHub Action which recompiles the project each time the repository is updated, achieving the same level of automation as the Read the Docs solution. One advantage of using GitHub Actions is that they provide more free computation resources compared to Read the Docs. For more information, see the relevant page in the [Jupyter Book docs](https://jupyterbook.org/publish/gh-pages.html).

## Contributing updates to this template

Projects based on this template are likely to introduce new improvements and fixes to achieve their goals. When possible, we encourage users to propose these improvements as a pull-request to this template repository so that new projects may benefit. However, since a project's code base will have, by design, diverged from *Sinn-full*, this requires more care than usual for upstream contributions. (*Sinn-full* does not want your project-specific stuff ;-) ). The basic idea is to fork a separate copy, and backport the changes into this fork; for reference, the basic steps are documented below.

:::{Note}  
Because this process involves transferring code between diverged code bases, it will invariably require some manual intervention.  
:::

- Step 1 is to fork the *Sinn-full* repository (not “Use this template”). When you make changes to this fork, GitHub will propose to create a pull-request (PR).

- Step 2 is to port the desired changes from the project (say “myproject”) back to *Sinn-full*. There are two ways to do this.

  + *Easy way*: Update the *Sinn-full* files directly, by editing them and/or copying over changes from the project files. Once finished, commit and push the changes.

    This is conceptually simple, but of course one must take care not to forget any desired changes (e.g. that extra import you added in another file). It works best for small changes and fixes.
    
  + *Hard way*: Use `git format-patch` to create a set of patches from your project, and `git am` to apply them to Sinn-full.
  
    This uses some rather advanced git commands, and if there are a lot of patches can require some effort to fix merge conflicts. The main benefit is that because the process is based on the project's commit history, it is less likely to forget changes. It also splits the change into logical commits, which can be easier to understand. Below is a suggested procedure; note that at intermediate steps the repository will be in an unclean state, so it is highly recommend to first understand the required git commands and especially how to reverse them.
    
    Recommendation: A useful pattern is to start the message of all commits in a project which should eventually be backport with the string `[backport]`.
    
    1. Create a new branch “patch-<date>”
    2. `git rebase -i <commit>`, where `<commit>` is the hash of the earliest commit to be included in the set of patches
    3. Use `drop`, `squash`, `fixup` to remove irrelevant commits and streamline the commit history.
       Note that patch conflicts can be onerous to resolve, so in general, having fewer commits will save time.
       Having a consistent tag identifying commits to be backported or squashed can greatly simplify this process.
    4. `git format-patch` will create a set of patch files starting with 0001, 0002, etc.
       Move these patch files to */tmp/patches*
    5. The patch files will include paths with the project name; we need to replace it with “sinnfull”.
       To do this
       - Create a new directory */tmp/new-patches*.
       - Apply the following sed command to replace all instances of “myproject” with “sinnfull” (take care not to accidentally replace other substrings)
    
             for file in `ls /tmp/patches`; do sed s:myproject:sinnfull:g /tmp/patches/$file > /tmp/new-patches/$file; done
             
    6. `git am --whitespace=warn --reject /tmp/new-patches/*`
       - `--whitespace=warn` is suggested because Markdown files may use two trailing spaces to indicate a newline.
       - `--reject` tells git to apply the patch hunks that succeeded, so that you only need to fix those that failed.
       - The application of some of the patches will generally fail when the file they are applied to has also changed. This will generate .rej files listing the unapplied changes:
         + Go through all .rej files and apply changes manually.
         + As changes are applied, delete the .rej file
         + When there are no .rej files left, stage the files (`git add`)
         + `git am --continue`
         + Repeat until all patches are applied
       - If you need to abort the process, you may want to use `git am --quit` instead of `git am --abort` to keep the changes that have already been applied.
       
- Step 3 is the usual procedure for a PR: push the changes to your forked Sinn-full repo, and open a PR via GitHub.


**Everything below this line may be used as a starting point for a project's README**

---

# Sinn-full project

{> Short project description <}

Based on the [Sinn-full](https://github.com/alcrene/sinnvoll) project template.

## Installation

If you are on \*nix, the following should suffice:

1. clone this repository onto your machine.

Then `cd` to the repository's directory and execute

% NB: List numbers currently restart at 1. because of a bug in MyST: https://github.com/executablebooks/MyST-Parser/issues/482
2.
       bash install.sh
3.
       smttask init

If on Windows, clone the repository, then set up a conda environment as you normally would. You can use the description of *install.sh* below as guidelines. Don't forgot to do run `smttask init` once the package is installed.

The *install.sh* script does the following:

- Propose a default name and location for your project, based on its directory name.
- Automatically find and merge environment files based on some hard-coded heuristics.
  + This allows the use of extra *\*-local.yaml*, *\*-cluster.yaml* files to define dependencies which are platform-specific.
- Ensure the environment is properly isolated from system packages (see [below](#Dependencies)).
- Install any additional packages specified in a *requirements.txt* files.
  + In most cases this is not necessary, since _pip_ dependencies (even from version control repositories) can be specified [directly in the environment file](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually).
- Make the *conda-forge* channel the default for any future packages.
- If a *lib* folder exists, install any packages it contains.
  + This is especially useful if your project depends on another which is also being developed. Simply place a symbolic link under *lib* pointing to the source directory for that package, and it will always be up to date.
- Install the code in this repository as a package within the new environment as a development install.
  + I.e., execute `pip install -e .`
- Add a hook so that if you use R and [reticulate](https://rstudio.github.io/reticulate/), it will use the this projet's environment rather than the system Python.

### Installing the Jupyter extensions

In many cases, conda will install the Jupyter extensions for you. However, if you are using an older version of JupyterLab, or if you run the Jupyter server in a separate environment, you may have to install them separately; the following will install the extensions used in the provided notebooks.

```bash
conda activate [jupyter environment name]
conda install jupyter jupyterlab widgetsnbextension nodejs conda-forge::jupytext
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install @bokeh/jupyter_bokeh
jupyter lab build
conda deactivate
```

> Adapt as appropriate if you have installed Jupyter in a virtualenv rather than a conda environment.

> The most recent conda packages include both the Python packages and the Jupyter Lab extensions. So you may not need the lines starting with `jupyter`.

## Literate programming with Jupytext

To make it easier to browse and adapt the template code files, most of them are in fact Jupyter Notebooks exported with [Jupytext](https://jupytext.readthedocs.io/). This allows inline-documentation with formatted mathematics, while still remaining version control friendly.

To view the formatted version of a file, navigate to the folder containing it in Jupyter, right-click the file and select “Open as Notebook”. To save a synchronized notebook (useful to avoid recomputing figures), open the [command palette](https://jupyterlab.readthedocs.io/en/stable/user/commands.html) and select “Pair Notebook with ipynb document”.

### Inlining test code

To inline code which should only be executed when a module is opened as a notebook, add an `if __name__ == "__main__:"` guard at the top of the cell, and indent the code so it is under the `if` clause. This is used throughout to provide illustrative or test code in notebooks without making them slow to import into other notebooks.

:::{hint}
Recent versions of Jupytext also allow similar functionality via [cell tags](https://jupytext.readthedocs.io/en/latest/formats.html?highlight=tag#active-and-inactive-cells).
:::

## Code organization

Subpackage (folder) names follow the following convention:

- *nouns* contain library code. This is code that might be usable as-is in another project.
  + [*data*](./sinnfull/data/index)  
    Objects which define a consistent API for interacting with on-disk and synthetic data.
  + [*models*](./sinnfull/models/index)  
    Both dynamical and stationary models.
    **This is where to add new models.**
  + [*diagnostics*](./sinnfull/diagnostics/__init__)  
    Functions to help diagnose fitting issues.
  + [*viz*](./sinnfull/viz/index) (visualization)  
    Visualization functions.

- *verbs* contain notebook files which are task-specific. These files are generally continuously modified and not added to version-control until they reach an 'archive' state. If a particular analysis is expected to be repeated, it may be distilled to a template notebook, which would then be included in version control.
  + Git's [`--skip-worktree` option](https://compiledsuccessfully.dev/git-skip-worktree/) may be useful to keep templates of these files without polluting your version control.
  + [*diagnose*](./sinnfull/diagnose/index)  
    Sample scripts for diagnosing fitting issues.
  + [*workflows*](./sinnfull/workflows/index)  
    Creation and execution of sequences of tasks. In its simplest form, a task creation script modifies one or more values from a default parameter file, and creates the associated task.
    **This is where to create and run inference tasks.**
  + [*view*](./sinnfull/view/index)  
    Exploring and viewing results of analysis tasks.

## Misc

### Syncing recorded runs with a remote server

All runs are recorded by _smttask_, and therefore _Sumatra_, into an SQLite database. A script is provided to synchronize remote and local run databases; simply copy _smtsync__template.sh_ to _smtsync.sh_ and change the values in the signposted lines. Although minimal and only tested on Linux, this shell script is specifically designed to be tolerant against changes on the server. In other words, it can be run on a local machine while the server is still running tasks.

### Removing Jupytext version information from script files

By default, when exporting to a script, Jupytext includes its version number in the metadata block. If multiple machines with different Jupytext version are used to edit files, this can create unnecessary file differences for git. The script _dont_save_jupytext_version.sh_ can be used to add a configuration option which removes this line from the metada.

Alternatively, that option can be set by editing the script files directly, by adding

```
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
```

under the `jupytext` heading.
