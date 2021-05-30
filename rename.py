"""
Execute this in a new project to give it a new name.
"""

import os
import shutil
from pathlib import Path
import argparse

# TODO:
#
# - Try to obtain the repo URL with git
# - Replace sinn-full URLs with the one that was obtained
# - Add option, or ask during rename, for project name to put in _config.
# - Add option, or ask during rename, for author
#     + _config
#     + setup.py

root = Path(__file__).parent

parser = argparse.ArgumentParser(
    description="Rename project\n"
                f"Starting from {root.absolute()}, recurse into subdirectories and "
                "files and replace the old project name by the new one, "
                "both in their content and in their name.")
parser.add_argument("new", nargs="*",
    help="The new project name.\n"
         "If multiple strings should be substituted, provide each separated by "
         "a space. In this case OLD should also be a list of the same length.")
parser.add_argument("--old", nargs="*", default=["sinnfull", "sinn-full"],
    help="The old project name.\nAll instances of this name will be replaced "
         "by NEW, both in file names and in their content.\n"
         "WARNING: The OLD name must be entirely unique, such that it "
         "does not appear in any substrings, since those will also be "
         "replaced. This assumption is safe for a vanilla sinn-full project.")

args = parser.parse_args()
old_names = args.old
new_names = args.new
if len(new_names) != len(old_names):
    if len(new_names) == 1:
        new_names = new_names*len(old_names)
    else:
        raise ValueError("Different number of NEW and OLD strings.\n"
                         f"NEW: {new_names}\nOLD: {old_names}")

# Add case variations to replace list
# (E.g. all-lowercase version of `old` may be used in identifiers)
# NB: Variations must come after originals, in case the match an original
# (e.g. `old` could already be all lowercase)
for old, new in zip(old_names[:], new_names[:]):
    if old.lower() not in old_names:
        old_names.append(old.lower())
        new_names.append(new.lower())
    if old.upper() not in old_names:
        old_names.append(old.upper())
        new_names.append(new.upper())
    if old.capitalize() not in old_names:
        old_names.append(old.capitalize())
        new_names.append(new.capitalize())
                         
readable_extensions = ['.py', '.ipynb', '.md', '.rst', '.yml', '.yaml', '.txt']
caching_dirs = ['.ipynb_checkpoints', '.cache', '.sinn.graphcache']
    # These directories contain caches. They are possibly invalidated
    # by the rename, so just delete them.

# Alternatively to `topdown=False`, we could modify `dirnames` in place to reflect the renamed directories
# This would allow skipping caching folders
for dirpath, dirnames, filenames in os.walk(root, topdown=False):
    dirpath = Path(dirpath)

    for dirname in dirnames:
        if dirname in caching_dirs:
            # Delete caching directories
            shutil.rmtree(dirpath/dirname)
        else:
            # Move files if they include the project.
            for old, new in zip(old_names, new_names):
                if old in dirname:
                    new_dirname = dirname.replace(old, new)
                    old_path = dirpath/dirname
                    old_path.rename(dirpath/new_dirname)

    for fname in filenames:
        file = dirpath/fname
        if file.suffix in readable_extensions:
            # Replace contents of files
            # Also replace all-lowercase version of `old`, which may be used in identifiers
            # NB: Do the lowercase replacement 2nd, in case `old` is all lowercase
            for old, new in zip(old_names, new_names):
                file.write_text(file.read_text().replace(old, new))
        # Move files if they include the project.
        for old, new in zip(old_names, new_names):
            if old in fname:
                new_fname = fname.replace(old, new)
                old_path = dirpath/fname
                old_path.rename(dirpath/new_fname)
