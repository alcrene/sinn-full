import os
from pathlib import Path
from importlib import import_module

from mackelab_toolbox.utils import stabledigest
from sinnfull.parameters import ParameterSet
from sinnfull.tags import TaggedCollection

"""
Scan subdirectories, find all optimization parameter sets
The model name, as determined from the containing directory, and the file name
are automatically added as tags to each found object.
"""

# NB: This is almost a carbon copy of sinnfull.models._scandir.

# Directories to skip when scanning for model elements
skip_dirs = []

# The subdirectory under which all parameter sets are placed, relative to the
# directory containing this file. Names of subdirs of `pset_dir` are assumed to
# correspond to model names.
pset_dir = "paramsets"

# Placeholders for the scan. Are replaced by TaggedCollections below
paramsets = {}
pset_tags = {}
    # Using dicts avoids having copies of the same object, since we key
    # with (model name, obj name)

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
        if isinstance(o, ParameterSet):
            add_pset(o, {model_name, nm})
        else:
            # Not a type we want to place in a collection
            continue

root = Path(__file__).parent/pset_dir
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


paramsets = TaggedCollection((paramsets[k], pset_tags[k]) for k in paramsets)
paramsets.append((str(root), {"basepath"}))
    # Add basepath for all paramsets, which is needed to load paramsets
    # with references
