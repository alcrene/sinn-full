# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (sinnfull)
#     language: python
#     name: sinnfull
# ---

# %% [markdown]
# # Inspection of the gradients / updates
#
# The optimization critically relies on the correct computation of likelihood gradients. There are a lot of ways this can go wrong, ranging from indexing errors, to incorrect specification of dependent variables. Thankfully this is managed by the library layer, so user code should not introduce these kinds of issues.
#
# The flip side of this is that if there *are* errors in the gradients, one likely needs to dive deeper and open up the library to find them.

# %% [markdown]
# ---
# ## Initialization
#
# > Important difference compared to the usual notebook initialization: addition of the line
# > ```python
# > sinnfull.diagnostics.set(True)
# > ```

# %%
import sinnfull
sinnfull.setup('theano')
import sinnfull.optim
sinnfull.diagnostics.set(True)
#import sinn
#sinn.config.trust_all_inputs = True  # Allow deserialization of arbitrary function
#import smttask
#smttask.config.record = False
#smttask.config.load_project(sinnfull.projectdir)

# %%
import logging
logging.basicConfig()

# %%
import numpy as np
import theano_shim as shim
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm.auto import tqdm
sns.set(rc={'font.sans-serif': ['CMU Bright', 'sans-serif'],
            'font.serif': ['Computer Modern Roman', 'serif'],
            'savefig.pad_inches': 0,                          # Don't change size when saving
            'figure.constrained_layout.use': True,            # Make sure there is space for labels
            'figure.figsize': [5.0, 3.0]
           })

# %%
from sinnfull.diagnostics.utils import (
    RecordData, set_to_zero,
    print_gradient, print_data, parse_theano_print_output)

# %%
# Load the record we want to use for our test
#rec_data = load_record('20200916-162115_4e31')
rec_data = RecordData('20210327-185932_1f6db8', ηhist='Itilde')


# %% [markdown]
#     # Workaround because runs were made before Recorders were fully unpackable
#     from mackelab_toolbox.typing import Array
#     from sinnfull.optim.recorders import LogpRecorder, ΘRecorder
#     import sinnfull.diagnostics.utils
#     kwargs = rec_data.logp.dict()
#     kwargs['values'] = [float(Array.validate(v)) for v in kwargs['values']]
#     rec_data.logp = LogpRecorder(**kwargs)
#     rec_data.Θ = ΘRecorder(**rec_data.Θ.dict())
#     sinnfull.diagnostics.utils.Θ = rec_data.Θ

# %% [markdown]
# Initialize params & hists with fit values at the optimization step of interest.
# This may be e.g. the last point of the fit, or the last point before a divergence.

# %%
rec_data.set_to_step(973)

# %% [markdown]
# If we are investigating a point where the optimizer broke, chances are that we its state wasn't recorded at exactly the point we want. Thus we need to choose a point before the break, and finish the iterations in the notebook

# %% [markdown]
# Before doing so, we reattach the recorders to the optimizer, and set them to record at every step.
# For even higher recording resolution, we could use a `DiagnosticRecorder`, which records on each parameter or latent update (rather than at the end of each pass).

# %%
optimizer = rec_data.optimizer

# %% [markdown]
# **TODO:** Attaching recorders to optimizers is being deprecated. Just call `ready` & `record` within the fit loop.

# %%
recorders = [rec_data.logp, rec_data.Θ, rec_data.latents]
for recorder in recorders:
    recorder.interval=2  # Hack to avoid warning
    recorder.interval_scaling='linear'
    recorder.interval=1
for recorder in recorders:
    optimizer.add_recorder(recorder)

# %% [markdown]
# Now we iterate up to the point of interest

# %%
for i in tqdm(range(optimizer.stepi, 1424)):
    optimizer.step()

# %%
#rec_data.logp.drop_step(1021)

# %%
plt.plot(rec_data.logp.steps, rec_data.logp.values)
plt.ylim(4700, 5100)

# %%
rec_data.logp.steps[-1]

# %%
rec_data.logp.values[-2]

# %%
plt.plot(rec_data.Θ.steps, np.array(rec_data.Θ.logτtilde)[:,0])

# %%
plt.plot(rec_data.Θ.steps, np.array(rec_data.Θ.logτtilde)[:,1])

# %%
rec_data.Θ.keys

# %%
plt.plot(rec_data.Θ.steps, np.array(rec_data.Θ._Wtilde_transp_stickbreaking__)[:,0])

# %%
plt.plot(rec_data.latents[-2]['Itilde'][:,0])

# %%
plt.plot(rec_data.latents[-2]['Itilde'][:,1])

# %%
rec_data.latents.steps.index(1000)

# %%
plt.plot(rec_data.latents[30]['Itilde'][:,1])

# %%
Itilde_T_evol = [Itilde[-1] for Itilde in rec_data.latents.Itilde]
Itilde_Tm1_evol = [Itilde[-2] for Itilde in rec_data.latents.Itilde]

# %%
model = rec_data.model
optimizer = rec_data.optimizer
λη = rec_data.λη
data = rec_data.data

# %% [markdown]
# ---

# %%
from collections import namedtuple
from sinnfull.viewing.record_store_viewer import FitCollection

# %%
rsview = sinnfull.diagnostics.utils.rsview
#rsview = rsview.filter.label(rec_data.record.label).list
#rsview = rsview.filter.reason('ground truth').filter.after(20210222).list
rsview = rsview.filter.label("20210226-024317_353910").list  # Ground truth

# %%
rsview.aslist()

# %%
rsview.η_curves().cols(2)

# %% [markdown]
# ---
# ## Evaluation of gradients using introspection functions
#
# We can use the introspection functions for the latent gradients, `full_gradη` and `sliced_gradη`. These are created when `sinnfull.diagnostic_hooks` is set to `True`, as we did above.

# %%
full_gradη = rec_data.full_gradη
sliced_gradη = rec_data.sliced_gradη

# %%
set_to_step(-1)

# %%
print_data(b=model.tnidx.plain-7, Kηb=4, Kηr=4)

# %%
print_gradient(b=model.tnidx.plain-7, Kηb=4, Kηr=4)

# %% [markdown]
# We plot the increment $-λ^η \cdot g^η_T$ at each time step. Alongside, we also plot $\tilde{I}_T$ – if the problem is a too large learning rate causing chaotic dynamics (by overshooting), we should see these two quantities anti-correlated.

# %%
gradη_T_evol = []
b=model.tnidx.plain-7; Kηb=8; Kηr=0
for i in range(len(rec_data.latents.steps)):
    set_to_step(i)
    gradη_T_evol.append( full_gradη(b, Kηb, Kηr)[-1] )

# %%
fig, ax = plt.subplots()
twinax = ax.twinx()
ax.plot(rec_data.latents.steps, np.array(Itilde_T_evol)[:,1]);
twinax.plot(rec_data.latents.steps, np.array(gradη_T_evol)[:,1], color=sns.color_palette()[1]);
ax.set_ylabel("$-λ^η \cdot g^η_T$")
twinax.set_ylabel("$\\tilde{I}_T$")
ax.set_xlabel("iterative step");

# %% [markdown]
# ---
# ## Performing additional fit iterations in the notebook
# Useful for
# - Testing changes to the optimizer
# - Triggering [inserted print statements](#Inserting-“print”-statements)
#
# Including the last point for the recorded fit can help identify issues, but may also raise red-herrings if the code has changed (i.e. been fixed) since the fit was done.

# %%
full_gradη = rec_data.full_gradη
sliced_gradη = rec_data.sliced_gradη

# %%
rec_data.full_gradη

# %%
set_to_step(-1)
step = rec_data.latents.steps[-1]
steps = []
gradηT = []
ItildeT = []

# %%
# Set to True if SGDOptimizer implements the Δt rescaling described above,
# and we are investigating the rightmost gradient
gradients_rescaled_by_Δt = True
Δt = model.dt.magnitude

# %%
## If print statements were added, it can be useful to reduce the number of
## batches per pass
#optimizer.fit_hyperparams['Nηb'] = 5

# %%
# The slice for which we evaluate the gradient
b=model.tnidx.plain-7; Kηb=8; Kηr=0

# %%
for i in range(50):
    optimizer.step()
    step += 1
    steps.append(step)
    gradηT.append(full_gradη(b, Kηb, Kηr)[-1])
    ItildeT.append(model.Itilde.data[-1])

# %%
fig, ax = plt.subplots()
colors = sns.color_palette()
twinax = ax.twinx()
if gradients_rescaled_by_Δt:
    rescale_str = "Δt \cdot "
    ax.plot(steps, (-Δt * λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
else:
    rescale_str = ""
    ax.plot(steps, (-λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
ax.set_ylabel(ax.lines[-1].get_label(), color=colors[0])
twinax.plot(steps, np.array(ItildeT)[:,1], color=colors[1]);
twinax.set_ylabel("$\\tilde{I}_T$", color=colors[1])
ax.yaxis.set_tick_params(color=colors[0], labelcolor=colors[0])
twinax.yaxis.set_tick_params(color=colors[1], labelcolor=colors[1])
ax.set_xlabel("iterative step");

# %% [markdown]
# Note that because of the rescaling after each pass, it may seem that the calculated increment $λ^η\cdot g^η_T$ has nothing to do with the *actual* increment of $I_T$ between two time steps $s$ and $s+1$:

# %%
fig, ax = plt.subplots()
colors = sns.color_palette()
twinax = ax.twinx()
if gradients_rescaled_by_Δt:
    rescale_str = "Δt \cdot "
    ax.plot(steps, (-Δt * λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
else:
    rescale_str = ""
    ax.plot(steps, (-λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
ax.set_ylabel(ax.lines[-1].get_label(), color=colors[0])
twinax.plot(steps[:-1], np.diff(np.array(ItildeT)[:,1]), color=sns.color_palette()[1]);
twinax.set_ylabel("$\\tilde{I}_T^{s+1} - \\tilde{I}_T^{s}$", color=colors[1])
ax.yaxis.set_tick_params(color=colors[0], labelcolor=colors[0])
twinax.yaxis.set_tick_params(color=colors[1], labelcolor=colors[1])
ax.set_xlabel("iterative step");

# %% [markdown]
# However, if we extract the value just before and after an increment, within the same pass (e.g. by inserting print statements), then the updates are as expected. The reason for the jumps, therefore, is the rescaling of $\tilde{I}$ and $\tilde{W}$ after each update to remove degeneracies. This rescaling can also explains the counter-intuitive correlation of the increments with $\tilde{I}$.

# %% [markdown]
# (Abbreviated code:)
# ```python
# s = """[[-2.24486842  0.44919974]
#  [-2.20250582  0.36285316]
#  [-2.27249193  0.39877222]]__str__ SHIM - I_T (before)
# [ 6.13224297e-05 -5.68013095e-07]__str__ SHIM - Δη_T
# [[-2.24299925  0.45154501]
#  [-2.20492344  0.3654928 ]
#  [-2.27243061  0.39877165]]__str__ SHIM - I_T (after)
# ...
# """
# ```

# %%
true_Δη = parse_theano_print_output(s, filter_str='Δη_T')
I_T_sm1 = parse_theano_print_output(s, filter_str="before")
I_T = parse_theano_print_output(s, filter_str="after")

fig, ax = plt.subplots()
ax.set_title("($\\tilde{I}_T$ obtained via print functions)")
colors = sns.color_palette()
twinax = ax.twinx()
if gradients_rescaled_by_Δt:
    rescale_str = "Δt \cdot "
    ax.plot(steps, (-Δt * λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
else:
    rescale_str = ""
    ax.plot(steps, (-λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
ax.set_ylabel(ax.lines[-1].get_label(), color=colors[0])
twinax.plot(steps[1:], (I_T - I_T_sm1)[1:,1], color=sns.color_palette()[1]);
twinax.set_ylabel("$\\tilde{I}_T^{s} - \\tilde{I}_T^{s-1}$", color=colors[1])
ax.yaxis.set_tick_params(color=colors[0], labelcolor=colors[0])
twinax.yaxis.set_tick_params(color=colors[1], labelcolor=colors[1])
ax.set_xlabel("iterative step");

# %% [markdown]
# ---
# ## Inserting “print” statements
#
# To interrogate the optimizer more deeply, we can start inserting print statements in the optimization routine. Then re-running the notebook will print these during the iteration steps above
#
# > Note that print statements in a static computational graph work [somewhat differently](http://www.deeplearning.net/software/theano/library/printing.html) than what you might expect. The `print` function provided by *theano_shim* is a convenience wrapper around Theano's printing function.
#
# > Remove any print statements before running any actual fits – they add up to a lot of CPU time, and will flood your console with messages.
#
# For example, in `optimization.SGDOptimizer.default_latent_optimizer`, this is the section which computes the rightmost gradient:
#
# ```python
# latent_updates['rightmost'] = {}
# for h, slc in K_slices['rightmost'].items():
#   Δt = getattr(h.dt, 'magnitude', h.dt)
#   inc = -λη[h]*gη[h][slc]
#   latent_updates['rightmost'][h._num_data] = shim.inc_subtensor(
#      h._num_data[slc], inc)
# ```
#
# To print the exact increment applied during optimization, change the line
#
# ```python
#   inc = -λη[h]*gη[h][slc]
# ```
# to
# ```python
#   inc = shim.print(-λη[h]*gη[h][slc], message("λη*gη_rightmost")
# ```

# %% [markdown]
# The `inspect_utils` module includes the function `parse_theano_print_outputs`, which may be useful to parse multiple lines of printed output back into an array. Use copy-paste to assign the output to a string variable, and pass that to the function.
# It's definitely a hack though, so check the result for correctness.

# %% [markdown]
# (Abbreviated code:)
# ```python
# s = """[2.19956135e-05 9.28919204e-05]__str__ DEBUG - Δη_T
# [4.95852071e-05 2.13593916e-05]__str__ DEBUG - Δη_T
# ...
# """
# ```

# %%
true_Δη = parse_theano_print_output(s, filter_str='Δη_T')

# %%
fig, ax = plt.subplots()
colors = sns.color_palette()
if gradients_rescaled_by_Δt:
    rescale_str = "Δt \cdot "
    ax.plot(steps, (-Δt * λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
else:
    rescale_str = ""
    ax.plot(steps, (-λη*np.array(gradηT))[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$")
ax.plot(steps, true_Δη[:,1], label=f"$-{rescale_str}λ^η \cdot g^η_T$ (optimizer)")
ax.set_xlabel("iterative step");
ax.legend();
