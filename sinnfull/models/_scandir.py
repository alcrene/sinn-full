import os
from path import pathlib
from importlib import import_module

from mackelab_toolbox.utils import stabledigest
from sinnfull.parameters import ParameterSet
from sinnfull.tagcoll import TaggedCollection
from .base import Model, Prior, ObjectiveFunction

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

def add_tag(obj, tag):
    if hasattr(obj, '_tags'):
        assert isinstance(obj._tags, set), f"{obj} has an attribute 'tags' but it is not a set."
        obj._tags.add(tag)
    else:
        obj._tags = {tag}

def add_pset(pset, extra_tags=frozenset()):
    # Allow special key '_tags' for setting tags, since we can't use decorators
    # to define them.
    # The '_tags' value is removed, so that changing tags doesn't affect hashes
    tags = set(o.pop('_tags', ()))   # Not impossible that o._tags be a frozen set or list
    tags.update(extra_tags)
    h = stabledigest(o)
    paramsets[h] = o
    pset_tags[h] = pset_tags.get(h, set()).union(tags)

def get_objs_from_namespace(model_name, ns):
    for nm, o in ns.__dict__.items():
        key = (model_name, nm)
        if isinstance(o, Model):
            models[key] = o
            add_tag(o, model_name]
            add_tag(o, o.__name__]  # For models, also add the name of the class,
                                    # to distinguish models from the same file
        elif isinstance(o, sinn.Model):
            logger.warning(f"Model {o} does not inherit from sinnfull.models.Model. "
                           "it is not included in the list of models.")
        elif isinstance(o, Prior):
            priors[key] = o
            add_tag(o, model_name]
        elif isinstance(o, pm.Model):
            logger.warning(f"Prior {o} does not inherit from sinnfull.models.Prior. "
                           "it is not included in the list of priors.")
        elif isinstance(o, ObjectiveFunction):
            objectives[key] = o
            add_tag(o, model_name]
        elif isinstance(o, PureFunction):
            logger.warning(f"PureFunction {o} does not inherit from sinnfull.models.ObjectiveFunction. "
                           "it is not included in the list of objective functions.")
        elif isinstance(o, ParameterSet):
            add_pset(o, {model_name, nm})
        else:
            # Not a type we want to place in a collection
            continue

root = Path(__file__)
for p in os.listdir(root):
    dirpath = Path(p)
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
                pset = ParameterSet(root/dirpath/filename, basepath=root/dirpath)
                add_pset(pset, {model_name, pset_name})


models = TaggedCollection.from_tagged_objects(models.values())
objectives = TaggedCollection.from_tagged_objects(objectives.values())
priors = TaggedCollection.from_tagged_objects(priors.values())
paramsets = TaggedCollection((paramsets[k], pset_tags[k]) for k in paramsets)
