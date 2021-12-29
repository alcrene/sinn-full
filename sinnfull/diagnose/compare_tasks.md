---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: md:myst,py:percent
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (sinn-full)
  language: python
  name: sinn-full
---

# Comparing task files

Compare two [_task files_](../workflows/Generate%20tasks.py) that should be the same but aren't.

Common reasons for this to happen:

- Values in the serialization with undefined order.
- A random seed is not set.

```{code-cell} ipython3
import json
```

```{code-cell} ipython3
taskfile1 = "/path/to/project/sinnfull/workflows/tasklist/OptimizeModel__6e091e77ff__nsteps_5000.taskdesc.json"
taskfile2 = "/path/to/project/sinnfull/workflows/tasklist/OptimizeModel__244934ad8f__nsteps_5000.taskdesc.json"
```

```{code-cell} ipython3
with open(taskfile1) as f:
    json1 = f.read()
with open(taskfile2) as f:
    json2 = f.read()
```

Find characters that differ and print the context around them.

```{code-cell} ipython3
quiet = False
for i, (c1, c2) in enumerate(zip(json1, json2)):
    if c1 != c2 and not quiet:
        print(f"------------- char {i} ----------------")
        print(json1[i-10:i+50])
        print(json2[i-10:i+50])
        quiet = True
    elif quiet and json1[i-5:i+1] == json2[i-5:i+1]:
        quiet = False
```

Print the sequence(s) of nesting levels that lead to a particular key.

```{code-cell} ipython3
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
```

```{code-cell} ipython3
data1 = json.loads(json1)
```

```{code-cell} ipython3
print_parents(data1, "μtilde")
```
