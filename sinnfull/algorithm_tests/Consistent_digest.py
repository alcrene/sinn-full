# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
# ---

# # Test digest consistency
#
# Recompute and print the digest from the latest recorded Task.
# Use this to confirm that different machines compute consistent digests,
# by running the script on each machine and comparing the output.
#
# (Requires there to be at least one recorded Task in the Sumatra record storey.)
#
# FIXME: Create the task within the script:
#
#   1) Doesn't require a Sumatra recordstore.
#   2) Will not accidentally indicate consistency just because an array isn't present.
#   3) Instantaneous because an optimizer is not uselessly compiled.
#     
# OR: Allow scanning the record store, computing digests for all/multiple records ?

import sinnfull
sinnfull.setup('theano')

from sinnfull import projectdir

import numpy as np

import smttask

records = smttask.view.RecordList()

task = smttask.Task.from_desc(records.latest.parameters)

print(task.digest)
