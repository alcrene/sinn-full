#! /bin/sh

# Each user will require slightly different configurations. The recommendation is to only
# add this template file to version control, and to let each user define their own sync paths.

# This script synchronizes two Sumatra project record stores.
# Typical use case is to update a local copy with new entries produced on an HPC resource.
# The script uses Unison to perform the synchronization over the network, and therefore
# inherits the strong guarantees of Unison with regards to leaving files in a consistent state.

# To describe synchronization, we define three Sumatra record stores:
# - A: the local record store
# - B: the remote record store
# - B': a local copy of the remote record store
# Synchronization proceeds in the following manner:
# 1. unison mirrors B locally to B'
# 2. Sumatra syncs A and B'. Any records only present in one are copied to the other.
#   NOTE: This means that records are never deleted.
# 3. unison attempts to sync B' with B. If B has changed since step 1, the sync
#    is aborted and the script restarts from 1. This is repeated up to five
#    times, until a successful sync back with B.

# Remarks
# - Synchronization allows for changes to occur both on the local and remote record stores.
# - Synchronization is tolerant to the REMOTE changing during the synchronization
#   (at worst, the remote record store will not contain new records from the local record store)
# - However, it is assumed that no new records are being created on the LOCAL machine.
# - DELETIONS are not sync'ed. If a record is deleted on one record store, it will be added
#   back on the next synchronization. To remove a record, it must be deleted from both record stores.

# Requirements
# - The same version of Unison must be installed on both the local and remote machines.

# -------------------------- CONFIGURATION CONSTANTS -----------------------------------
# Values to fill are {> bracketed <}.

CONDAENV="{> ENVNAME  <}"     # Name of conda environment used to run this project
PROJECTNAME="{> PROJECTNAME <}"  # Name of project within Sumatra
LOCALPROJECTDIR="{> $HOME/PATH/TO/PROJECT <}"                   # Project dirs must be in the
REMOTEPROJECTDIR="{> ssh://USER@IP-OR-URL/PATH/TO/PROJECT <}"   # format expected by Unison
MIRROREDPROJECTDIR="$PROJECTNAME-smt"

# ---------------------------------------------------------------------------------------

mirroronly=no
overwritelocal=no
if [ $# -gt 0 ]; then
    # There are positional parameters to parse
    if [ "$1" = "--mirroronly" -o "$1" = "--mirror" ]; then
        mirroronly=yes
    elif [ "$1" = "--force" -o "$1" = "--force-local" -o "$1" = "--overwrite" ]; then
        overwritelocal=yes
        read -n 1 -p "This will overwrite the local record store with the one on the remote. Continue ? (Y/n) " confirm
        if [ "$confirm" = "n" -o "$confirm" = "N" ]; then
            echo ""
            echo "Aborting."
            exit
        fi
    elif [ "$1" = "-h" -o "$1" = "--help" ]; then
        echo "smtsync [--mirroronly]"
        echo "  -h           ~ Print this documentation"
        echo "  --help"
        echo ""
        echo "  --mirror     ~ Do not sync: only mirror the server to the local machine."
        echo "  --mirroronly   This can be useful if we want a perfect copy of the record store "
        echo "                 on the server. Note that only the local mirror is modified; to "
        echo "                 replace the local record store with the mirror, the mirror must "
        echo "                 be manually copied over the local store. This obviously destroys "
        echo "                 the local store, which is why it must be done manually."
        echo ""
      	echo "  --force      ~ Like --mirroronly, but also performs the copy on top of "
      	echo "  --force-local  the local record store. A confirmation message is shown before "
      	echo "                 proceeding, since this operation can lead to data loss."
        exit
    else
        echo "Unrecognized parameter '$1'."
        exit
    fi
fi

# Make `conda activate` work in shell script  (https://github.com/conda/conda/issues/7980#issuecomment-492784093)
eval "$(conda shell.bash hook)"

# Activate conda environment (Do it early, so we know immediately if environment does not exist)
conda activate $CONDAENV

# Make temporary location for mirrored remote
mkdir -p ~/tmp
cd ~/tmp
mkdir -p "$MIRROREDPROJECTDIR"
    # The mirrored directly is left in place, so that on subsequent syncs, only differences
    # need to be transmitted.

# Three steps: 1) Mirror server DB to local temp dir
#              2) Sync with local DB
#              3) Push changes back to server
# The third step is only permitted if there have been no server changes in the mean time.
# If there have been changes, we repeat from 1 again, up to 5 times.

# Loop to allow multiple tries
n=0
until [ "$n" -ge 5 ]; do
    # Use Unison to mirror the server DB into the local temp dir
    # We use Unison instead of rsync here so that when we attempt to send the updates back to the server,
    # it will tell us if the server DB has changed in the mean time
    echo "Mirroring remote Sumatra project $PROJECTNAME..."
    unison "$REMOTEPROJECTDIR/.smt" "$MIRROREDPROJECTDIR" -force "$REMOTEPROJECTDIR/.smt" -batch -ui text

    if [ "$mirroronly" = "yes" ]; then
        echo "Server record store mirrored to location $(pwd)/$MIRROREDPROJECTDIR."
        exit
    elif [ "$overwritelocal" = "yes" ]; then
        cp "$MIRROREDPROJECTDIR/"* "$LOCALPROJECTDIR/.smt/"
        # TODO: Check that copy succeeded ? (either return val or time stamp of new files)
        echo "Local record store at location $LOCALPROJECTDIR/.smt was overwritten with the remote record store."
        exit
    fi

    # Synchronize local Sumatra DB with mirrored copy of the server DB
    # REMARK: sync expects the actual DB (in the default config, the 'records' file)
    echo "Synchronizing local and mirrored Sumatra records..."
    smt sync "$LOCALPROJECTDIR/.smt/records" "$MIRROREDPROJECTDIR/records"

    # # The next line is intended for testing: it gives one time to inspect the sync result before pushing, and to change the server DB to test change detection
    # read -p "Sumatra sync complete. Press any key to push changes to the server." -n1

    # Use Unison again to push the changes to the server;
    echo "Pushing updated records back to server..."
    unison "$REMOTEPROJECTDIR/.smt" "$MIRROREDPROJECTDIR" -auto -noupdate "$MIRROREDPROJECTDIR" -batch -ui text

    if [ "$?" -eq 0 ]; then break; fi

    n=$((n+1))
    if [ "$n" -ge 5 ]; then
        echo "Unable to push updated records to the server: it seems to constantly change."
    else
        echo "Server records have changed; trying again."
    fi
done
