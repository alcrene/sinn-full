#! /bin/sh

# Two intended use-cases:
# - Ensure all code files have a metadata header, so that cell tags are saved.
#   (See e.g. https://github.com/mwouts/jupytext/issues/714#issuecomment-759964756)
# - Prevent Jupytext from saving its version number in the metadata header.

# Saving the jupytext version number is annoying when there are multiple
# machines involved. This can be deactivated on a per-notebook basis by
# executing (see https://github.com/mwouts/jupytext/issues/416):
#
#     jupytext <notebook-filename>.ipynb --update-metadata '{"jupytext":{"notebook_metadata_filter":"-jupytext.text_representation.jupytext_version"}}'

# The command below applies this recursively within a directory, skipping
# checkpoint files.

# IMPORTANT: This must be executed within the environment containing the
# Jupyter installation, or at least one where Jupytext is installed.

# NB: This is hardcoded to run from its the location of the file at the top of
# the project.

find sinnfull \( \( -name "*.py" -o -name "*.ipynb" \) -not -name "__init__.py" -not -name "_*.py" \) -print0 | while read -d $'\0' file; do
    if [[ ! "$file" == *".ipynb_checkpoints"* ]]; then
        jupytext "$file" --update-metadata '{"jupytext":{"notebook_metadata_filter":"-jupytext.text_representation.jupytext_version"}}';
    fi;
done
