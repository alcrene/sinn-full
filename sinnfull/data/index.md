# Data Accessors

The purpose of _data accessors_ is to provide a consistent interface for sampling datasets and retrieving relevant metadata. “Relevant metadata” may include e.g. subject name, or the parameters used to produce a synthetic trial. This allows the same optimization algorithm to be used with different data sources, both experimental and synthetic.

The interface is composed of two classes:

class `Trial`
  ~ All information relevant to a single trial: metadata, path to disk, methods for loading data from disk, etc.
  ~ Defines the trial attributes used to uniquely identify trials.

class `DataAccessor`
  ~ Collection of `Trial` objects. Includes methods for constructing the list of trials.  
    Importantly, collections only contains metadata – the data themselves are never loaded all at once.
  ~ A `load` method. Takes an instance of `Trial` and loads the associated data.

The expected attributes and methods for each of these classes are defined in [base.py](./base.py). The other modules implement classes specific to a particular dataset. For example, if data from an experiment is stored in a hierarchy of directories as `subject/date/condition_A_trial_n`, then a `DataAccessor` can be written to extract the relevant information form the directory and filenames, and construct `Trial` objects accordingly. If another experiment stores all data in a single HDF5 file, another `DataAccessor` may be written to do the same with that data.

The _sinn-full_ template provides one complete and one template implementation:

[Synthetic accessor](./synthetic.py)
  ~ A generic accessor for synthetic data which, instead of loading data from disk, generates it from the same kinds of [models](../models) used for inference. This is useful for initial testing, as well as _in silico_ experiments.
    + `DataAccessor` sets the model.
    + Each `Trial` stores the model _params_ and random simulation _seed_ as metadata.

[Accessor template](./template.py)
  ~ When writing a new accessor, this can be used as a starting point.

One of the goals of the `DataAccessor` was to simply the definition of loaders for _custom data storage solutions_, also known as “piles of data files roughly organized into directories” – a common approach for smaller scale experimental data. As such, there are mechanisms for:

- Lazily loading data one file at a time (important since even small data sets can occupy many GB);
- Defining the file naming scheme for data files, and finding all matching files within a directory hierarchy.
- Defining a separate scheme for metadata files, which store information stored across experiments.
- Extracting trial information from file names.

## Getting started

The [`SyntheticDataAccessor`](./synthetic.py) is fully functional and can be used as-is for _in silico_ experiments.

For recorded date, a custom DataAccessor needs to be defined. Begin by copying the [template](./template.py) to a new file, and then modify as needed. Instructions are included in the template file.

:::{Note}  
The intent and design of data accessors (along with that of the [sampling module](../sampling.ipynb)) is very close to that found in PyTorch's [data utilities](https://pytorch.org/docs/stable/data.html). Specically, our `DataAccessor` is analagous to PyTorch's `Dataset`, and our `SegmentSampler` merges PyTorch's `Sampler` and `DataLoader` types. In the future we may restructure to reproduce the PyTorch API, to save users from learning yet another interface.
:::
