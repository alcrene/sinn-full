# -*- coding: utf-8 -*-
# %% [markdown]
# # Comparing task files
#
# Compare two [_task files_](../workflows/Generate%20tasks.py) that should be the same but aren't.
#
# Common reasons for this to happen:
#
# - Values in the serialization with undefined order.
# - A random seed is not set.

# %%
import json

# %%
taskfile1 = "../workflows/tasklist/n5000/OptimizeModel__543fa9d7eb__nsteps_5000.taskdesc.json"
taskfile2 = "../workflows/tasklist/n5000/OptimizeModel__d5cb560448__nsteps_5000.taskdesc.json"

# %%
with open(taskfile1) as f:
    json1 = f.read()
with open(taskfile2) as f:
    json2 = f.read()

# %% [markdown]
# Find characters that differ and print the context around them.

# %%
quiet = False
for i, (c1, c2) in enumerate(zip(json1, json2)):
    if c1 != c2 and not quiet:
        print(f"------------- char {i} ----------------")
        print(json1[i-10:i+50])
        print(json2[i-10:i+50])
        quiet = True
    elif quiet and c1 == c2:
        quiet = False


# %% [markdown]
# Print the sequence(s) of nesting levels that lead to a particular key.

# %%
def print_parents(d: dict, s: str, parents=[]):
    # print all entries at the same nested level together
    # to keep the output easier to read
    for k in d:
        if s in k:
            print(parents + [k])
    for k, v in d.items():
        if isinstance(v, dict):
            print_parents(v, s, parents+[k])
        elif s in str(v):
            print(parents + [k], "->", v)

# %%
data1 = json.loads(json1)

# %%
print_parents(data1, "μtilde")
