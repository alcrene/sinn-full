import os
from pathlib import Path
from importlib import import_module
from collections.abc import Callable

import logging
logger = logging.getLogger(__name__)

import pymc3 as pm
from smttask.typing import PureFunction
from mackelab_toolbox.utils import stabledigest
import sinn

from sinnfull.parameters import ParameterSet
from sinnfull.tags import TaggedCollection, TagDecorator
from .base import Model, Prior, ObjectiveFunction, tag

"""
Scan subdirectories, find all modelling objects and place into collections.
The model name, as determined from the directory, is automatically added as
a tag to each found object. This
a) seems convenient / desirable and
b) ensures that the '_tags' attribute exists for TaggedCollection.
"""

# Directories to skip when scanning for model elements
skip_dirs = []

# Placeholders for the scan. Are replaced by TaggedCollections below
models = {}
objectives = {}
priors = {}
paramsets = {}
pset_tags = {}
    # Using dicts avoids having copies of the same object, since we key
    # with (model name, obj name)

def isstrictsubclass(cls, class_or_tuple):
    class_tuple = class_or_tuple
    if isinstance(class_tuple, type):
        class_tuple = (class_tuple,)
    return (isinstance(cls, type) and issubclass(cls, class_or_tuple)
            and all(cls is not T for T in class_tuple))

def add_pset(pset, extra_tags=frozenset()):
    # Allow special key '_tags' for setting tags, since we can't use decorators
    # to define them.
    # The '_tags' value is removed, so that changing tags doesn't affect hashes
    tags = set(pset.pop('_tags', ()))   # Not impossible that o._tags be a frozen set or list
    if not tags:
        tags = set(pset.pop('tags', ()))
    tags.update(extra_tags)
    h = stabledigest(pset)
    paramsets[h] = pset
    pset_tags[h] = pset_tags.get(h, set()).union(tags)
    # Add _tags back, to allow overspecifying tag filters
    pset._tags = pset_tags[h]

def get_objs_from_namespace(model_name, ns):
    for nm, o in ns.__dict__.items():
        key = (model_name, nm)
        if isstrictsubclass(o, Model):
            models[key] = o
            tag(model_name)(o)
            tag(o.__name__)(o)  # For models, also add the name of the class,
                                    # to distinguish models from the same file
        elif isinstance(o, type) and issubclass(o, sinn.Model) and o is not Model:
            logger.warning(f"Model {o} does not inherit from sinnfull.models.Model. "
                           "it is not included in the list of models.")
        elif isinstance(o, ObjectiveFunction):
            objectives[key] = o
            tag(model_name)(o)
        elif isinstance(o, PureFunction):
            logger.warning(f"PureFunction {o} does not inherit from sinnfull.models.ObjectiveFunction. "
                           "it is not included in the list of objective functions.")
        elif isinstance(o, ParameterSet):
            add_pset(o, {model_name, nm})
        elif (isinstance(o, Callable) and hasattr(o, '_tags')
              and not isinstance(o, TagDecorator)):
            # HACK: Since we use factory functions for priors, we assume that
            # all priors have at least one tag and keep all callables that
            # define '_tags'.
            # TODO?: Discard any prior without a tag matching at least one model
            priors[key] = o
            tag(model_name)(o)
        else:
            # Not a type we want to place in a collection
            continue

root = Path(__file__).parent
for p in os.listdir(root):
    dirpath = root/Path(p)
    # Only import directories if they don't start with '_' or '.', and are not explicitely excluded
    if dirpath.is_dir() and p[0] not in "_." and p not in skip_dirs:
        model_name = p
        # Get anything defined in __init__.py, if it exists
        ns = import_module(f"sinnfull.models.{model_name}")
        get_objs_from_namespace(model_name, ns)  # Fills globals
        for filename in os.listdir(dirpath):
            if filename.endswith(".py"):
                # Import each Python file, in case it wasn't imported by __init__ (or there is no __init__)
                modname = Path(filename).stem
                ns = import_module(f"sinnfull.models.{model_name}.{modname}")
                get_objs_from_namespace(model_name, ns)  # Fills globals
            elif filename.endswith(".paramset"):
                # Read each parameter set
                # We add the file name as a tag, since that seems like it could be useful
                pset_name = Path(filename).stem
                pset = ParameterSet(dirpath/filename, basepath=dirpath)
                add_pset(pset, {model_name, pset_name})

models = TaggedCollection.from_tagged_objects(models.values(), tag_attr='_tags')
objectives = TaggedCollection.from_tagged_objects(objectives.values(), tag_attr='tags')
priors = TaggedCollection.from_tagged_objects(priors.values(), tag_attr='_tags')
paramsets = TaggedCollection((paramsets[k], pset_tags[k]) for k in paramsets)
