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
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # Inspecting the effect of hyperparameter
#
# As is typical with machine learning, the results of a fit is sensitive to hyperparameters. Possible symptoms are
#
# - Fits are erratic / parameter fits seem to overshoot. \
#   (This is easier to spot with synthetic data where ground truth is known.)
#     + The learning rate is likely too high.
# - The $\log L$ seems to flatten out, then drops again.
#     + There could be an issue with the learning rate schedule, or, if you are using an optimizer with state (e.g. Adam), the hyperparameters setting this schedule.
#     + This might also be related to a too high learning rate.
#         + For instance, in the particular case of Adam, the only state parameter is the quantity $\frac{\sqrt{1-(1-β_2)^i}}{1 - (1-β_1)^i} \in [0, 1]$, which simply depresses the learning rate over the initial steps.
#

# %%
import sinnfull
sinnfull.setup('theano')
import sinnfull.optim
sinnfull.optim.diagnostic_hooks = True
import sinn
sinn.config.trust_all_inputs = True  # Allow deserialization of arbitrary function
import smttask
smttask.config.record = False
smttask.config.load_project(sinnfull.projectdir)

# %%
import logging
logging.basicConfig()

# %%
import numpy as np
import theano_shim as shim
from tqdm.auto import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sinnfull.viz import plot_fit_dynamics  # Sets sns defaults

# %%
import sinnfull.diagnostics
from sinnfull.diagnostics.utils import load_record, set_to_step, set_to_zero
sinnfull.diagnostics.enable()

# %%
from sinnfull.optim.recorders import LogpRecorder, ΘRecorder

# %% [markdown]
# ---
# ## Re-running the optimizer produces different results
#
# We first plot the dynamics of the log likelihood and the parameters.
# Then we reset to a particular optimization step and rerun the optimizer for a few steps. If the optimizer has internal state (e.g. a learning schedule), then the new traces won't be identical to the old, but if diverging traces are suddenly converging again, then we have good evidence that optimizer state is the problem.

# %%
# Load the record we want to use for our test
rec_data = load_record('20200924-235015_a0fa')

# %%
model = rec_data.model
optimizer = rec_data.optimizer
latents = rec_data.latents
logp = rec_data.logp
Θ = rec_data.Θ
λη = rec_data.λη
data = rec_data.data

# %%
step_idx = 14

# %% [markdown]
# Dashed lines indicate the interval where we re-run optimizer.

# %%
fig, axes = plt.subplots(1, 3)
fig.set_figwidth(2*mpl.rcParams['figure.figsize'][0])

ax = axes[0]
ax.plot(rec_data.logp.steps, rec_data.logp.values)
ax.axvline(latents.steps[step_idx], color='#333333', linestyle='--');
ax.axvline(latents.steps[step_idx+1], color='#888888', linestyle='--');
ax.set_title("$\\log L$");
ax.set_xlabel("iteration step")

ax = axes[1]
set_to_step(step_idx)
ax.plot(*optimizer.model.Itilde.trace);
ax.set_title(f"$\\tilde{{I}}$ @ step {latents.steps[step_idx]}");

ax = axes[2]
set_to_step(step_idx+1)
ax.plot(*optimizer.model.Itilde.trace);
ax.set_title(f"$\\tilde{{I}}$ @ step {latents.steps[step_idx+1]}");

# %% [markdown]
# Integrate between two integration steps and record the log likelihood and parameter evolution. We set Recorders to record at every iteration.

# %% [markdown]
# **TODO:** Attaching recorders to optimizers is being deprecated. Just call `ready` & `record` within the fit loop.

# %%
optimizer.add_recorder(LogpRecorder(interval=1, interval_scaling='linear'))
optimizer.add_recorder(ΘRecorder(optimizer, interval=1, interval_scaling='linear'))

# %%
set_to_step(step_idx)
n_steps = latents.steps[step_idx+1]-latents.steps[step_idx]
print(f"Iterating the optimizer {n_steps} more steps...")
for i in tqdm(range(n_steps)):
    optimizer.step()

# %%
fig, axes = plt.subplots(1, 2)
fig.set_figwidth(1.5*mpl.rcParams['figure.figsize'][0])

ax = axes[0]
ax.plot(rec_data.logp.steps, rec_data.logp.values, label='original run')
ax.axvline(latents.steps[step_idx], color='#333333', linestyle='--');
ax.axvline(latents.steps[step_idx+1], color='#888888', linestyle='--');

rec = optimizer.recorders['log L']
rec_data_logp_idx = np.searchsorted(rec_data.logp.steps, rec_data.latents.steps[step_idx])
c = sns.color_palette()[1]
ax.plot(rec_data.logp.steps[:rec_data_logp_idx+1], rec_data.logp.values[:rec_data_logp_idx+1], c=c)
ax.plot(latents.steps[step_idx]+np.array(rec.steps), rec.values, c=c, label='new run');
ax.set_title("$\\log L$");
ax.set_xlabel("iteration step")
ax.legend()
ax.set_ylim(ymin=-3500)

ax = axes[1]
ax.plot(*optimizer.model.Itilde.trace);
ax.set_title(f"$\\tilde{{I}}$ @ step {latents.steps[step_idx]} (new)");

# %% [markdown]
# > **Remark**: If there is a break in the curve for the “new run” log likelihood, the likely issue is that not all changing parameters are being recorded. To check whether this is the case, execute the code below: if the two parameter sets differ, some parameters are not being recorded and therefore not being reset by `set_to_step`.

# %%
from IPython.display import display

# %%
set_to_step(step_idx)
print(f"--- Model parameters @ step {latents.steps[step_idx]} --- ")
display(model.params.get_values())
for i in range(3):
    optimizer.step()
set_to_step(step_idx)
print(f"--- Model parameters after iterating 3 steps and resetting to step {latents.steps[step_idx]} --- ")
display(model.params.get_values())

# %%
fig, axes = plot_fit_dynamics({'Θ': Θ})
fig.suptitle("Fit dynamics");

# %% [markdown]
# ---
# ## Check whether latent updates destroy optimization later in the trace
#
# Because latent dynamics are updated in batches, optimizations on earlier batches can in theory worsen the fit at points beyond a batch's the end point. A few things mitigate this:
#
# - Choosing $N^η_b$ large enough that batches have large overlaps.
# - With the hyper parameter $T^η_r > 0$, data points are added to the end of the batch when computing the likelihood. This points are not optimized, thus the latest optimized point is kept consistent with subsequent data points.
# - The point-wise likelihood is generally only affected by the points immediately before and after. So a single unlikely jump should not have a disproportionate impact on the likelihood as a whole.
#
# Still, these are not hard guarantees, and it makes sense to check that updates at late times are kept consistent with those at earlier times. One way to do this is to attach `DiagnosticRecorder` to the optimizer, to record say $\log L(\tilde{I}_{k:K_end})$, for a few different values of $k$.

# %%
from sinnfull.diagnostics.utils import partial_logps_recorder
from sinnfull.optim.recorders import logL_recorder

# %%
Kηb = shim.eval(optimizer.Kηb)
Kηr = shim.eval(optimizer.Kηr)
K_list = [i*(Kηb+Kηr) for i in range(1, 4)]
k_list = [optimizer.model.time.tnidx.plain - K for K in K_list]
partial_logp_recorder = PartialLogpRecorder(
    start_k=k_list, partial_lengths=K_list
)

# %% [markdown]
# **TODO:** Attaching recorders to optimizers is being deprecated. Just call `ready` & `record` within the fit loop.

# %%
optimizer.add_recorder(partial_logp_recorder)

# %%
optimizer.add_recorder(logL_recorder)

# %%
n_steps = latents.steps[step_idx+1]-latents.steps[step_idx]
print(f"Iterating the optimizer {n_steps} more steps...")
for i in range(n_steps):
    optimizer.step()

# %%
plt.plot(*optimizer.model.Itilde.trace);

# %%
optimizer.diagnostic_recorders['batch_logp'].values

# %%
rec = optimizer.recorders['log L']
plt.plot(rec.steps, rec.values);
