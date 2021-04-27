#!/bin/sh

set -e

# TODO: Convert to a python script. This would
#       a) make it much to maintain
#       b) make it windows-compatible

# These commands create a new conda environment from scratch, grabbing
# the most current version for each package.
# To *reproduce* an existing environment, export
#     conda list --export > env-freeze.yml
# and follow the instructions at the top of the file

## Hard-coded values. Use these to override values inferred from the directory structure
# (Leaving blank lets the script infer values from the directory structure
ENVNAME=
PROJECTROOT=
ENVDIR=
ENVYML=

# https://stackoverflow.com/a/21188136
get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}
#https://stackoverflow.com/a/2924755
bold=$(tput bold)
normal=$(tput sgr0)

# Change to the script's directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "Switching to directory $DIR"
cd "$DIR"

INSTALL_LOCATION="local"
# # To set INSTALL_LOCATION based on the computer hostname, uncomment below
# # and adjust the `if` conditions.
# # Determine if we are on a cluster
# # These are hard-coded tests based on the hostname
# # They determine two things: whether to do a CPU or GPU install, and whether
# # to install ipykernel so the environment can be executed from Jupyter
# if [[ `hostname` == "login" ]]; then
#   echo "You shouldn't be installing from the cluster's login node."
#   echo "Aborting."
#   exit
# elif [[ `hostname` == "cluster-name" ]]; then
#   INSTALL_LOCATION="cluster"
# else
#   # Anything else we presume is a local install
#   INSTALL_LOCATION="local"
# fi

# Make `conda activate` work in shell script  (https://github.com/conda/conda/issues/7980#issuecomment-492784093)
eval "$(conda shell.bash hook)"

# Set the name of the conda environment – this is what will be typed
# to activate conda: `conda activate $ENVNAME`
# Here we set it to the name of the directory containing this file,
# in lowercase for easier typing.
# If the parent's name is 'code', the name of its parent is used instead.
namedir=$DIR
if [ ! $ENVNAME ]; then  # Skip if ENVNAME is hard-coded in script
    ENVNAME=`basename "$namedir"`
    while [ "$ENVNAME" == "code" ]; do
        namedir=`dirname "$namedir"`
        ENVNAME=`basename "$namedir"`
    done
    ORIGENVNAME=$ENVNAME
    ENVNAME="${ENVNAME,,}"  # Use shell parameter expansion to make lowercase
fi

# Make `conda activate` work in shell script  (https://github.com/conda/conda/issues/7980#issuecomment-492784093)
eval "$(conda shell.bash hook)"

## Ensure we don't clobber an existing environment
# Get the list of current conda environments
envlist="$(conda env list | tr "*" " ")"

for env in $envlist; do
  if [ "$env" == "$ENVNAME" ]; then
# if [[ "$envlist" == *"$ENVNAME "* ]]; then  #
    echo "Environment '$ENVNAME' is already installed."
    doenvinstall=n
    break
  else
    doenvinstall=y
  fi
done

# Determine the root directory (PROJECTROOT) for the project
# PROJECTROOT should not be in the repo: it's a place to put project-related
# stuff (like environments) which are not tracked
# Here we set it to the parent folder of the repository

# Move up until we find the root directory for the repo
if [ ! $PROJECTROOT ]; then  # Skip if PROJECTROOT is hard-coded in script
  cd "$namedir"
  while [ -d .git ]; do
    if [ "$(pwd)" == "/" ]; then
      echo "Cannot find a parent which isn't a git repository."
      echo "Rather than relying on this script, you specify the PROJECTROOT by editing the file install.sh."
      exit
    fi
    cd ..
  done
  PROJECTROOT="$(readlink -f .)"
    # (readlink -f turns relative path into absolute)
  # Change cur dir back to DIR
  cd $DIR
fi

# Set the environments directory
if [ ! $ENVDIR ]; then  # Skip if ENVDIR is hard-coded in script
    ENVDIR="$PROJECTROOT/envs"
fi

# Determine the name of the environment YAML file
# We take the first file which matches, in order
#   - $ENVNAME.yml
#   - environment.yml
#   - env.yml
if [ $ENVYML ]; then
    1  # pass
elif [ -f "$ENVNAME.yaml" ]; then
    ENVYML="$ENVNAME.yaml"
elif [ -f "$ENVNAME.yml" ]; then
    ENVYML="$ENVNAME.yml"
elif [ -f "environment.yaml" ]; then
    ENVYML="environment.yaml"
elif [ -f "environment.yml" ]; then
    ENVYML="environment.yml"
elif [ -f "env.yaml" ]; then
    ENVYML="env.yaml"
elif [ -f "env.yml" ]; then
    ENVYML="env.yml"
else
    echo "Could not find an environment YAML file. Aborting."
    exit
fi

# Determine which conda env files to merge
# The found ENVYML is used as a stem for specialized files
# TODO: Use the found ENVYML filename as a stem for suffixes
env_files=$ENVYML
stem="${ENVYML%.*}"
ext="${1##*.}"
if [ $INSTALL_LOCATION == "local" ] && [ -e $stem-local.$ext ]; then
  env_files="$env_files $stem-local.$ext"
fi

# Verify with the user the auto-detected file names and paths

if [ $doenvinstall == "y" ]; then

  echo "You are about to install a conda environment with the following properties:"
  echo "  - Environment name:      $ENVNAME"
  echo "  - Environment location:  $ENVDIR"
  echo "  - YAML description file: $ENVYML"
  echo ""
  read -p "Proceed ? (Ctrl-C to abort) " -n 1


  ## Create conda environment ##

  # # Create and activate new environment
  # conda env create --prefix "$ENVDIR/$ENVNAME" --file "$ENVYML"

  # Merge the conda env files
  # We need 'conda-merge' for this, which we install in a throaway environment
  # Create a throaway environment in which to install conda-merge
  echo ""
  echo "${bold}Merging environment files...${normal}"
  python3 -m venv /tmp/tmp-python
  source /tmp/tmp-python/bin/activate
  pip install --upgrade pip wheel
  pip install conda-merge
  conda-merge $env_files > _merged_env.yaml
  deactivate
  rm -r /tmp/tmp-python

  # Create the new environment, using the merged environment file
  echo ""
  echo "${bold}Creating Conda environment...${normal}"
  if [ -e "$ENVDIR/$ENVNAME" ]; then
    echo ""
    echo "${bold}ERROR:${normal} The directory '$ENVDIR/$ENVNAME' already exists. If this is a conda "
    echo "environment, it is not known to your current conda installation. "
    echo "(One way this can happen is if the directory is copied from elsewhere.) "
    echo "The simplest resolution is probably to delete or move the directory "
    echo "and let conda install a fresh environment."
  fi
  conda env create --prefix "$ENVDIR/$ENVNAME" --file _merged_env.yaml

  # TODO: only append if not already in `conda env list`
  conda config --append envs_dirs "$ENVDIR"

  conda activate "$ENVDIR/$ENVNAME"

  # Add a pyvenv.cfg file prohibiting the use of user site packages.
  # (This is the default for normal Python venvs, but not Conda environments)
  # The pyvenv.cfg file must be placed one directory above the Python executable
  # (see https://docs.python.org/3/library/site.html)
  THISENVDIR="$( cd "$( dirname "$( dirname "$(which python)" )" )" && pwd )"
  echo "include-system-site-packages = false" >> $THISENVDIR/pyvenv.cfg
    # Unlikely that the pyvenv.cfg file exists, but append (instead of write) in case it does
    # NB: $THISENVDIR should be equivalent to $ENVDIR/$ENVNAME, but I prefer to
    #     explicitly go up one directory up rather than rely on Conda conventions

  # Install pip requirements
  # Since these may be sensitive to the installation order (especially if
  # they depend on git repos), the requirements files may be named
  # requirements-1.txt, requirements-2.txt, etc.
  if [ -e requirements*.txt ]; then
    for file in requirements*.txt; do
      echo ""
      echo "${bold}Installing additional dependencies from $file...${normal}"
      pip install -r "$file"
    done
  fi

else
  conda activate "$ENVDIR/$ENVNAME"

  echo ""
  echo "Detected YAML file: $ENVYML"
  echo "Environment installation was skipped because it is already installed."
  read -p "Would you like to update the environment based on the merged environment file (y/N) ? " -n 1 update_env
  if [[ $update_env =~ ^[Yy]$ ]]; then
    # TODO: Don’t repeat code from above
    # Merge the conda env files
    # We need 'conda-merge' for this, which we install in a throaway environment
    # Create a throaway environment in which to install conda-merge
    echo ""
    echo "${bold}Merging environment files...${normal}"
    python3 -m venv /tmp/tmp-python
    source /tmp/tmp-python/bin/activate
    pip install --upgrade pip
    pip install conda-merge
    conda-merge $env_files > _merged_env.yaml
    deactivate
    rm -r /tmp/tmp-python

    conda env update --file _merged_env.yaml
    if [ -e requirements*.txt ]; then
      for file in requirements*.txt; do
        echo ""
        echo "${bold}Installing additional dependencies from $file...${normal}"
        pip install -r $file
      done
    fi
  fi
fi

echo ""
read -p "Proceed with registering the kernel and installing local packages ? (Ctrl-C to abort) " -n 1

conda activate "$ENVDIR/$ENVNAME"

# Add the conda channels as defaults for this environment
# TODO: Read from env.yaml
# TODO: Only add those not already present
conda config --env --prepend channels conda-forge

# If `ipykernel` is in the '.yml' file, make the new kernel discoverable from Jupyter
if grep -q ipykernel _merged_env.yaml; then
    echo ""
    echo "${bold}Registering new environment as the IPython kernel \"$ENVNAME\".${normal}"
    python -m ipykernel install --user --name $ENVNAME --display-name "Python ($ORIGENVNAME)"
fi

cd "$DIR"
# Install any bundled library dependencies
if [ -d "lib" ] && [ -n "$(ls "lib")" ]; then
    # TODO: Set a flag and print this at the end
    echo ""
    echo "You have bundled dependencies in the 'lib' directory."
    echo "This is intended for development only; when deploying, specify all "
    echo "dependencies environment or requirements files"
    for package in $(ls lib); do
        pip install -e "lib/$package"
        if [ -d "src/$package" ]; then
          # Remove conda's source install, which we've just replaced
          yes | rm -r "src/$package"
        fi
    done
fi
# Delete conda's src directory if it's now empty
if [ -d src ]; then
  if [ -z "$(ls -A src)" ]; then  # See https://superuser.com/a/352290
    rm -r src
  fi
fi

## Install the code in this directory
if [ -e "setup.py" ]; then
    echo ""
    echo "${bold}Installing project code...${normal}"
    pip install -e .
fi

### Preparatory configuration for R
# This section creates a .Rprofile file which will point Python to the
# correct conda environment.
# This .Rprofile expects a file ~/.conda/condapath, which is created if
# necessary.

# TODO: Make .Rprofile optional

# Ensure the ~/.conda/condapath file is present
# We do it this way because conda installs itself in such a way that only login
# shells can see it, and R loads a non-login shell. Otherwise conda only works
# with one of a handful of hard-coded paths which R knows.
if [ ! -e "$HOME/.conda/condapath" ]; then
  condapath="$(which conda)"
  echo ""
  if [ $condapath ]; then
    echo "$condapath" > "$HOME/.conda/condapath"
    echo "Stored the path to the conda executable in \`~/.conda/condapath\`."
  else
    echo "Could not find the conda executable. The \`condapath\` file could not be created."
  fi
fi

# Create the .Rprofile file
if [ ! -e .Rprofile ]; then
cat >.Rprofile <<EOF
# ~/.conda/condapath is a text file with the path to conda, created by install.sh
condapath <- utils::read.table("~/.conda/condapath")
condapath <- as.vector(condapath\$V1[1])
library(reticulate)
reticulate::use_condaenv("$ENVNAME", conda=condapath, required=TRUE)
EOF
echo "Created an .Rprofile file. If you create an RStudio project in this directory, it will use this conda environment."
fi

# Remind user to set up smttask
echo ""
echo "Remember to complete the project setup by intializing smttask:"
echo ""
echo "\$ conda activate $ENVNAME"
echo "\$ smttask init"
echo ""
