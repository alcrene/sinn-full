# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python (sinn-full)
#     language: python
#     name: sinn-full
# ---

# %% [markdown]
# # Tests and examples for the Alternated Optimizer
#
# :::{caution}
#
# These tests _must_ be updated to reflect changes since they were written.
#
# They do _not_ currently work.
# :::

# %% [markdown]
# As an example and validation of the `AlternatedOptimizer`, we apply it below to fitting just the OU input model, treating the $I_k$ as observations and the $\tilde{I}_k$ as latents.^[We don't really need to treat the $\tilde{I}_k$ as latents if we only want to infer the model parameters, but it serves the illustration.]

# %% [markdown]
# ## Using the AlternatedOptimizer
#
#    1. Read-in the required model parameters and fitting hyperparameters from files.
#    2. Map the parameters from optimization to model space
#    3. Instantiate the model.
#    4. Instantiate the data accessor.
#       - This object is used to sample the data files, yielding a new data segment for each pass.
#         Importantly, data is loaded on demand, and released when it is no longer used, so
#         the data set size is only limited by disk space.
#    5. Define the *logp* function.
#    6. Instantiate the optimizer.
#    7. Call the optimizer's `compile_optimization_functions` method.
#       - This takes non-negligible computation time, which is why it isn't done
#         automatically during instantiation.
#         Also, there may be rare cases where we need to specify how to initialize
#         the model, which this method provides for.
#    8. Call ``step`` until the fit has converged.

# %%
if __name__ == "__main__":
    from sinnfull.optim.optimizers.alternated_sgd import *

    import itertools
    from sinnfull import ureg
    from sinnfull.parameters import ParameterSet
    from sinnfull.models import OUInput, TimeAxis
    from sinnfull.data import DataAccessor
    #from sinnfull.sampling import sample_baseline_segment
    from sinnfull.rng import get_fit_rng, get_sim_rng
    from sinnfull.optim import Recorder
    from mackelab_toolbox.optimizers import Adam


    ## Extract parameters from files ##
    from sinnfull import projectdir
    ParameterSet.default_basepath = projectdir/"sinnfull"

    params = ParameterSet("basic-wc-model.paramset")
    fit_hyperparams = ParameterSet("basic-wc-learning.paramset")
    fit_hyperparams.latents.???? *= 0.0001
    fit_units = fit_hyperparams.units
    params.remove_units(fit_units)
    fit_hyperparams.remove_units(fit_units)

    simrng = get_sim_rng(seed_key=(0,), exists_ok=True)  # Set exists_ok to False in production
    M    = params.dynamics.M
    N    = params.observations.N
    T    = fit_hyperparams.T
    ??t   = fit_hyperparams.??t
    time = TimeAxis(min=0, max=T, step=??t, unit=fit_units['[time]'])
    ??_optim = prior.sample((3,4), space='optim')
    # Or: read from file
    ??_model = OU.Parameters(**prior.backward_transform(??_optim))

    ## Instantiate the model
    input_model = OUInput(params=??_in, time=time, rng=simrng)

    ## Apply SGD mask to parameters to select which will be optimized ##
    #?? = prior.

    ## Instantiate data accessor (Not actually used in this example) ##
    #data = DataAccessor(sumatra_project="..", datadir="external/ExampleSzrData")
    # For this example, just generate some fake data from the model
    input_model.integrate('end', histories=[input_model.I])
    input_model.I.lock()  # Prevent from being cleared
    trueItilde = input_model.Itilde.copy()  # Save for comparing with inferred input
    trueI = input_model.I.data
    input_model.clear()

# %% [markdown]
# \begin{align}
#   l(??;??) &= \log P(I,\tilde{I},\tilde{I}_0|??) \\
#     &= \sum_{k=1}^K \biggl\{ \sum_{m=1}^M \log P(I^m_k|??,\tilde{I}_{k'\leq k},u_0,\tilde{I}_0) + \sum_{m=1}^{\tilde{M}} \log P(\tilde{I}_{k}|??,\tilde{I}_{k'< k},u_0,\tilde{I}_0) \biggr\} + \log P(\tilde{I}_0)\\
#     &= - \frac{K \tilde{M}}{2} \log \sqrt{2??} - K \sum_{m=1}^{\tilde{M}} \log \left( \tilde{??}^m \sqrt{\frac{2 ??t}{\tilde{??}^m}} \right) - \sum_{k=1}^K \sum_{m=1}^{\tilde{M}} \frac{\left(\tilde{I}^m_k - \tilde{I}^m_{k-1} - \frac{(\tilde{I}^m_{k-1} - \tilde{??}^m)??t}{\tilde{??}^m}\right)^2}{4 (\tilde{??}^m)^2 ??t / \tilde{??}^m} \notag \\
#     &\quad + \log P(\tilde{I}_0) (\#eq:log-l-ou-input) \\
#     &\quad - \sum_{k=1}^K \sum_{m=1}^M \left(\tilde{W}\cdot \tilde{I}^m_k - I^m_k\right)^2 \\
#     &=: \sum_{k=1}^K l_k(??;??) - d(\tilde{W}\cdot \tilde{I} - I) + \log P(u_0) + \log P(\tilde{I}_0)
# \end{align}
# (Note that we've added a squared error term to allow $\tilde{W}$ to be fit without the observation model.)

# %% [markdown]
# In the definitions below, notice how in `backward_logp`, we use square brackets for indexing. This is because in the backward pass, all histories should already be computed. Using square brackets ensures an error is raised if this is not the case.
#
# The difference between `accumulate` and `static_accumulate` is that the former evaluates at `cur_tidx + 1`, thus triggering symbolic updates, while the latter evaluates at `cur_tidx`, indicating to use the current value.
#
# > **Remark** Since the two functions are the same, we could instead have defined just a single `logp()` function and let `SGDOptimizer` wrap it with `accumulate` and `static_accumulate`.
#
# > **Remark II** There are two possible signatures for a ``logp``??function: \
#     - ``logp(k)`` \
#     - ``logp(self, k)`` \
#     When the second is used, the model is assigned to the `self` argument. This gives you access to the model namespace, without having to define ``logp`` within the model itself. (Conceivably one may want to try different objective functions with a model, and so defining requiring that ``logp`` be defined within the model is not ideal.)

# %%
if __name__ == "__main__":
    ## Define the logp function ##

    @input_model.accumulate
    def forward_logp(k):
        ??t = input_model.dt
        ??t = getattr(??t, 'magnitude', ??t)
        ??tilde=input_model.??tilde; ??tilde=input_model.??tilde; ??tilde=input_model.??tilde
        Wtilde=input_model.Wtilde
        Itilde=input_model.Itilde; I=input_model.I
        norm_Ik = shim.log(??tilde*shim.sqrt(2*??t/??tilde)).sum()
        gauss_Ik = ((Itilde(k) - Itilde(k-1) - (Itilde(k-1)-??tilde)*??t/??tilde)**2 / (4*??tilde**2*??t/??tilde)).sum()
        sqrerr = ((shim.dot(Wtilde, Itilde(k)) - I(k))**2).sum()
        return - norm_Ik - gauss_Ik - sqrerr

    @input_model.static_accumulate
    def backward_logp(k):
        ??t = input_model.dt
        ??t = getattr(??t, 'magnitude', ??t)
        ??tilde=input_model.??tilde; ??tilde=input_model.??tilde; ??tilde=input_model.??tilde
        Wtilde=input_model.Wtilde
        Itilde=input_model.Itilde; I=input_model.I
        norm_Ik = shim.log(??tilde*shim.sqrt(2*??t/??tilde)).sum()
        gauss_Ik = ((Itilde[k] - Itilde[k-1] - (Itilde[k-1]-??tilde)*??t/??tilde)**2 / (4*??tilde**2*??t/??tilde)).sum()
        sqrerr = ((shim.dot(Wtilde, Itilde[k]) - I[k])**2).sum()
        return - norm_Ik - gauss_Ik - sqrerr

# %% [markdown]
# On every iteration, we want to scale the learning rate with the stationary standard deviation.

# %%
if __name__ == "__main__":
    def update_hyper??(optimizer):
        # Remark: It is best not to define functions outside this one, because
        # if we serialize the optimizer, only `update_hyper??` will be included
        ???? = optimizer.orig_fit_hyperparams.latents.????
        stats = optimizer.model.stationary_stats()
        updates = {'latents': {'????': {}}}
        for h in optimizer.latent_hists:
            updates['latents']['????'][h] = ????*stats[h]['std'].get_value()
        return updates

# %%
if __name__ == "__main__":
    ## Instantiate the optimizer ##
    fitrng = get_fit_rng(seed_key=(1,), exists_ok=True)  # Set exists_ok to False in production
    optimizer = SGDOptimizer(
        model             =input_model,
        rng               =fitrng,
        #sample_data_segment=lambda: sample_baseline_segment(data, 'NA', t0=0, T=fit_hyperparams.T),
        data_segments     =itertools.repeat(((1,), slice(0,20), {'I': trueI})),
        observed_hists    =[input_model.I],
        latent_hists      =[input_model.Itilde],
        params            =??,
        fit_hyperparams   =fit_hyperparams,
        update_hyperparams=update_hyper??,
        logp_params       =forward_logp,
        logp_latents      =backward_logp
    )
    optimizer.compile_optimization_functions()

# %% [markdown]
# ## Single batch tests

# %%
if __name__ == "__main__":

    def test_??_updates(model, ??, k0=100, Kb=4, Kr=2, print_diff=False):
        model.clear()
        model.integrate(k0+Kr+Kb)
        correct_integration = (model.cur_tidx == k0+Kr+Kb)
        if print_diff:
            print(f"Before ?? update, current tidx was {model.cur_tidx} (should be {k0+Kr+Kb})",
                  f" ({'correct' if correct_integration else 'incorrect'})")
        before = {??:getattr(model.params, ??).get_value() for ?? in ??}
        optimizer.update_??(100, 4, 2)
        after = {??:getattr(model.params, ??).get_value() for ?? in ??}
        unchanged_params = [nm for nm in before if np.any(before[nm] == after[nm])]

        if model.cur_tidx != k0+Kr+Kb:
            correct_integration = False
        if print_diff:
            print(f"After ?? update, current tidx is {model.cur_tidx}   (should be {k0+Kb+Kr})",
                  f" ({'correct' if model.cur_tidx==k0+Kr+Kb else 'incorrect'})")
        if print_diff:
            print("The following parameters should have changed but didn't: ", unchanged_params)
        return unchanged_params, correct_integration

    def test_??_updates(latent_hist, upd_name, k0=100, Kb=4, Kr=2, print_diff=False):
        """
        Return a bool array comparing latent vars that have changed ot those
        that haven't.
        For a successful test, the returned array should be all True.
        :param latent_hist: A latent history modified by the functions in `update_??`
        :param upd_name: One of 'default', 'rightmost', 'leftmost'.
        :return: List of indices that were incorrectly updated or not updated
        """
        assert latent_hist.cur_tidx >= k0+Kb+Kr+2
        before = latent_hist.data
        optimizer.update_??[upd_name](k0, Kb, Kr);
        after = latent_hist.data

        #should_update = np.zeros(Kb+Kr+3, dtype=bool)
        if upd_name == 'default':
            should_update = np.arange(k0,k0+Kb)
            #should_update[2:2+Kb] = True
        elif upd_name == 'rightmost':
            should_update = np.arange(k0,k0+Kb+Kr)
            #should_update[2:] = True
        elif upd_name == 'leftmost':
            should_update = np.arange(k0+Kb)
            #should_update[:Kb] = True
        else:
            assert False

        diff = (before != after)
        updated = np.unique(np.where(diff)[0])
        corr_updates = [i for i in updated if i in should_update]
        if len(corr_updates) > 8:
            corr_updates = corr_updates[:7] + ['...']
        err_should_have_updated = [i for i in should_update if i not in updated]
        if len(err_should_have_updated) > 8:
            err_should_have_updated = err_should_have_updated[:7] + ['...']
        err_should_not_have_updated = [i for i in updated if i not in should_update]
        if len(err_should_not_have_updated) > 8:
            err_should_not_have_updated = err_should_not_have_updated[:7] + ['...']
        if print_diff:
            print("Correctly updated: ", corr_updates)
            print("Should have updated but didn't: ", err_should_have_updated)
            print("Should not have updated but did: ", err_should_not_have_updated)
        return err_should_have_updated + err_should_not_have_updated

# %%
if __name__ == "__main__":

    optimizer.model.Itilde.unlock()
    optimizer.model.clear()
    optimizer.model.integrate(0)
    assert optimizer.model.I is input_model.I
    assert input_model.Itilde is optimizer.model.Itilde
    assert input_model.I.locked
    assert input_model.I.cur_tidx == input_model.I.tnidx
    assert test_??_updates(input_model, ??, print_diff=True) == ([], True)

# %%
if __name__ == "__main__":
    optimizer.model.integrate('end')
    optimizer.model.I.lock()
    print('?? ??? default')
    assert not test_??_updates(input_model.Itilde, 'default', print_diff=True)
    print('?? ??? rightmost')
    assert not test_??_updates(input_model.Itilde, 'rightmost', print_diff=True)
    print('?? ??? leftmost')
    assert not test_??_updates(input_model.Itilde, 'leftmost', print_diff=True)


# %% [markdown]
# ## Instrospection helpers
#
# The definitions and functions below were useful in debugging this module.

# %%
def get_subs??(optimizer):
    return {optimizer.k.plain  : optimizer.model.curtidx_var,
            optimizer.K??b.plain: optimizer.K??b_symb,
            optimizer.K??r.plain: optimizer.K??r_symb}
def get_subs??(optimizer):
    return {optimizer.k.plain  : optimizer.model.curtidx_var,
            optimizer.K??b.plain: optimizer.K??b_symb,
            optimizer.K??r.plain: optimizer.K??r_symb}

if config.diagnostic_hooks:

    # Use this e.g. as: optimizer._compile_context.param_updates.eval(args??)
    args?? = {optimizer.model.curtidx_var: 100,
             optimizer.K??b_symb: 4,
             optimizer.K??r_symb: 2}
    args?? = {optimizer.model.curtidx_var: 100,
             optimizer.K??b_symb: 4,
             optimizer.K??r_symb: 2}
    subs?? = get_subs??()
    subs?? = get_subs??()

## Inspection of updates to ?? ##
def get_????(optimizer):
    "Updates to ?? (i.e. ????*grad??, where grad?? is filtered through Adam)"
    if not config.diagnostic_hooks:
        raise RuntimeError("`get_????` can only be called when `diagnostic_hooks` are enabled.")
    K??b_symb = optimizer.K??b_symb
    K??r_symb = optimizer.K??r_symb
    subs?? = get_subs??(optimizer)
    ????_graph = [optimizer._compile_context.param_upds[??] - ?? for ?? in optimizer.??]
    ???? = shim.graph.compile(inputs=(optimizer.model.curtidx_var, K??b_symb, K??r_symb),
                            outputs=shim.graph.clone(????_graph, replace=subs??),
                            on_unused_input='ignore')
    return ????

if config.diagnostic_hooks:
    ???? = get_????()

## Inspection of updates to ?? ##
def get_grad??(optimizer, upd_type, wrt_hist,
              batch_indices=False, full_grad??=False, sliced_grad??=False):
    """
    Three functions to aid the instrospection of the ?? update:

    - Output the time indices of the batch
    - Output the full gradient
    - Output the gradient, sliced to the batch window

    Set the argument flags for the desired arguments to ``True``.

    .. note:: This function returns the *gradient* of ??, whereas
       `get_????` returns the *finite difference* of ?? updates.

    These functions were useful in ensuring the sliced gradient
    aligned with the computed batch.

    Parameters
    ----------
    optimizer: Optimizer instance
    upd_type: 'default' | 'rightmost' | 'leftmost'
    wrt_hist: History instance
        The history with respect to which to compute the ?? gradient.
        If only the `batch_indices` are desired, can be set to ``None``.
    batch_indices
    full_grad??
    sliced_grad??: bool
        Flag parameters. Set to ``True`` to compile the corresponding
        inspection function.

    Returns
    -------
    tuple of functions
        Length is determined by the number of argument flags set to ``True``.
        The order of the returned functions is the same as in the signature,
        irrespective of the order in which argument flags are passed.
        Each function takes three arguments:
            tidx (model), K??b, K??r
    """
    if not config.diagnostic_hooks:
        raise RuntimeError("`get_grad??` can only be called when `diagnostic_hooks` are enabled.")
    model = optimizer.model
    g?? = optimizer._compile_context.g??[upd_type]
    K??b_symb = optimizer.K??b_symb
    K??r_symb = optimizer.K??r_symb
    subs?? = get_subs??(optimizer)

    ret_fns = []
    if batch_indices:
        ret_fns.append( shim.graph.compile(inputs=(optimizer.model.curtidx_var, K??b_symb, K??r_symb),
                                           outputs=shim.graph.clone(shim.arange(model.tnidx,symbolic=True)[slc],
                                                                    replace=subs??),
                                           on_unused_input='ignore')
                      )
    if full_grad??:
        ret_fns.append( shim.graph.compile(inputs=(optimizer.model.curtidx_var, K??b_symb, K??r_symb),
                                           outputs=shim.graph.clone(g??[wrt_hist],
                                                                    replace=subs??),
                                           on_unused_input='ignore')
                      )
    if sliced_grad??:
        slc = optimizer._compile_context.K_slices[upd_type][wrt_hist]
        ret_fns.append( shim.graph.compile(inputs=(optimizer.model.curtidx_var, K??b_symb, K??r_symb),
                                           outputs=shim.graph.clone(g??[wrt_hist][slc],
                                                                    replace=subs??),
                                           on_unused_input='ignore')
                      )
    return tuple(ret_fns)

# The following partial function avoid the need to unpack a length 1 result
def get_grad??_batch_indices(upd_type):
    return get_grad??(upd_type=upd_type, wrt_hist=None, batch_indices=True)
def get_grad??_full(upd_type, wrt_hist):
    return get_grad??(upd_type=upd_type, wrt_hist=wrt_hist, full_grad??=True)
def get_grad??_sliced(upd_type, wrt_hist):
    return get_grad??(upd_type=upd_type, wrt_hist=wrt_hist, sliced_grad??=True)

if config.diagnostic_hooks:
    batch_indices, full_grad??, sliced_grad?? = get_grad??(
        optimizer, upd_type='default', wrt_hist=input_model.Itilde,
        batch_indices=True, full_grad??=True, sliced_grad??=True)

# %%
if config.diagnostic_hooks:
    batch_indices(100,4,2)

    full_grad??(100,4,2)[99:108]  # One before, one after the window ??? these should be zero

    sliced_grad??(100,4,2)  # Should match the values from line above corresponding to listed batch indices

# %% [markdown]
# ## Full optimization tests

# %%
if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

# %% [markdown]
# Add recorders for the log likelihood, parameters and latent histories.
# To allow serialization, we define named functions rather than use lambdas.

# %%
if __name__ == "__main__":
    def get_logL(optimizer):
        return optimizer.logp()
    def get_??(optimizer):
        return tuple(shim.eval(??) for ?? in optimizer.??)
    def get_latents(optimizer):
        return tuple(h.get_data_trace(include_padding=True) for h in optimizer.latent_hists)
    optimizer.add_recorder(Recorder(name='log L', record=get_logL))
    optimizer.add_recorder(Recorder(name='??', record=get_??, keys=tuple(??.keys())))
    optimizer.add_recorder(Recorder(name='latents', record=get_latents,
                                    keys=[h.name for h in optimizer.latent_hists],
                                    interval=5))

# %%
if __name__ == "__main__":
    n_optimization_steps = 100  # 1000

    cur_i = optimizer.stepi
    optimizer.record()
    for i in range(cur_i+1,n_optimization_steps+1):
        optimizer.step(update_params=False)
    optimizer.record()
    print(f"step i = {cur_i}..{optimizer.stepi}")

# %% [markdown]
# #### Example 1: Inferring only the latent dynamics

# %% [markdown]
# With $W = \begin{pmatrix} 0.5&0.5 \\ 0.5&0.5 \end{pmatrix}$ and $?? = \begin{pmatrix} 6 \\ 0.4 \end{pmatrix}$, the dynamics of both $I^0$ and $I^1$ are dominated by $\tilde{I}^0$. Thus, we are able to infer $\tilde{I}^0$, but not $\tilde{I}^1$; in this case, the effect of the prediction error $\frac{\left(\tilde{I}^m_k - \tilde{I}^m_{k-1} - \frac{(\tilde{I}^m_{k-1} - \tilde{??}^m)??t}{\tilde{??}^m}\right)^2}{4 (\tilde{??}^m)^2 ??t / \tilde{??}^m}$ in Eq. \@ref(eq:log-l-ou-input) is to smooth the inferred trace $\tilde{I}^1$.

# %%
if __name__ == "__main__":
    fig, ax = plt.subplots(1,1)
    logp_recorder = optimizer.recorders['log L']
    ax.plot(logp_recorder.steps, logp_recorder.values);
    ax.set_yscale('symlog')
    if len(ax.get_yticks() < 2):
        # Automatic tick determination failed; display at least the bounds
        ax.set_yticks([min(logp_recorder.values), max(logp_recorder.values)])
    ax.set_title("log likelihood")
    ax.set_xlabel("# of passes")
    plt.show()

# %%
if __name__ == "__main__":

    latent_recorder = optimizer.recorders['latents']
    n_recs = len(latent_recorder.steps)
    rec_idcs = range(n_recs)[::n_recs//5]

    ploth = 2.5
    plotw = 4
    ncols = len(rec_idcs)
    nrows = input_model.Mtilde
    fig, axes = plt.subplots(nrows,ncols, squeeze=False, sharex=True, sharey='row', figsize=(plotw*ncols,ploth*nrows))

    xarr = trueItilde.get_time_array(include_padding=True)
    Itilde_target = trueItilde.get_data_trace(include_padding=True)
    for i, axes_comp in zip(rec_idcs, axes.T):
        Itilde_trace = latent_recorder.values[i][0]
        for m, ax in enumerate(axes_comp):
            ax.fill_between(xarr, np.zeros(len(xarr)), Itilde_target[:,m], color=sns.color_palette('pastel')[0])
            ax.plot(xarr, Itilde_trace[:,m])
        axes_comp[0].set_title(f"Step {latent_recorder.steps[i]}")
    for m, ax in enumerate(axes[:,0]):
        ax.set_ylabel(f"$\\tilde{{I}}^{m}$");
    plt.show()

# %%
