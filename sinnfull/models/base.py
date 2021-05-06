# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
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
# # Base classes for models, priors and objective functions
#
# Defines:
#
# - [`Param`](#param-type)
# - [`Model`](#models)
# - [`Prior`](#prior)
# - [`ObjectiveFunction`](#objectivefunction)

# %% tags=["remove-input"]
from __future__ import annotations

# %% tags=["hide-input"]
import functools
import itertools
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, Optional, Set, List, Dict
from pydantic import validate_arguments
import mackelab_toolbox.serialize as mtbserialize
from smttask.typing import PureFunction, CompositePureFunction
import operator
import numpy as np
import pymc3 as pm
import theano
import theano_shim as shim
import sinn

import abc
from numbers import Number
from pydantic import BaseModel, PrivateAttr, validator, root_validator
# from typing import FrozenSet
from sinn.utils.pydantic import add_exclude_mask
from sinn.models import ModelParams

import mackelab_toolbox.typing as mtbtyping
from mackelab_toolbox.typing import Array, Shared
from mackelab_toolbox.pymc_typing import PyMC_Model, PyMC_RV

from sinnfull.utils import TypeDict, add_to
from sinnfull.parameters import ParameterSet
from sinnfull.typing_ import IndexableNamespace
from sinnfull.rng import get_seedsequence
from sinnfull.tags import TagDecorator

# %%
__all__ = ["Param", "Model", "Prior",
           "ObjectiveFunction", "AccumulatedObjectiveFunction"]


# %% [markdown] tags=["remove-cell"]
# TODO
# - Remove Prior.logpt_replace
# - Fix Regularizer to work without this
# - Also validate tags added to ObjectiveFuncion after instantiation with
#   the `tag` decorator.

# %% [markdown]
# ## Param type
#
# `Param` is used to define model parameters that may be derived – i.e., variables defined with `Deterministic` in the prior. It extends `Shared` by additionally allowing PyMC random variables.

# %%
class ParamMeta:
    def __getitem__(self, key):
        if shim.config.library == 'numpy':
            return Shared[key]
        else:
            assert shim.config.library == 'theano'
            return Union[Shared[key], PyMC_RV]
Param=ParamMeta()

# %% [markdown]
# ## Models
#
# A small specialization of `~sinn.Model`, which adds support for defining _stationary statistics_.
#
# Subclasses should _define_ the following methods, if expressions for stationary statistics are available:
#
# - `_stationary_stats`
# - `stationary_dist`
#
# Subclasses should _use_ the following methods to access stationary statistics:
#
# - `stationary_stats` -> `sinn.models.ModelParams` (namespace of symbolic vars)
# - `stationary_stats_eval` -> `dict` (dict of real values)
# - `stationary_dist` -> `pymc3.Model`
#
# These latter methods take care of generic type casting, so that one only needs to implement the equations themselves in `_stationary_stats`. See the [Ornstein-Uhlenbeck](./OU/OU) model for an example.

# %% [markdown]
# :::{margin} Code
# `Model`: Stationary distribution
# :::

# %% tags=["remove-cell"]
from sinn.models import ModelMetaclass as BaseModelMetaclass
class ModelMetaclass(BaseModelMetaclass):
    def __getattr__(cls, attr):
        if hasattr(sinn.Model, '__getattr__'):
            try:
                return sinn.Model.__getattr__(cls, attr)
            except AttributeError:
                pass
        if attr != '_tags' and attr in getattr(cls, '_tags', set()):
            return cls
        else:
            raise AttributeError(f"{cls} does not define "
                                 f"the attribute '{attr}'.")


# %% tags=["hide-input"]
class Model(sinn.Model, metaclass=ModelMetaclass):
    _compiled_functions: dict=PrivateAttr(default_factory=lambda: {})

    @abc.abstractmethod
    def initialize(self, initializer: Union[None,str,tuple,dict]=None):
        """
        initializer:
            None: Defined by concrete model
            str:
                - 'stationary': Initialize from stationary distribution
                - concrete models may define other accepted values
            tuple: Initialize from stationary distribution with key `initializer`
            dict: Should have one entry for each initializable history and
                submodel. In composite models, can be used to specify different
                initializers for each submodel.

            With composite models, unles `initializer` is a dictionary, the
            same initializer is passed to each submodel.
        """
        raise NotImplementedError

    # Subclasses may use instance methods if necessary
    @classmethod
    def _stationary_stats(cls, params: ModelParams):
        """
        The model-specific `_stationary_stats` can assume that it always receives
        a ModelParams object (as opposed to a dict or IndexableNamespace)

        If expressions for the stationary statistics are not available, this function
        should raise `NotImplementedError`.

        Returns
        -------
        Dict[str,Dict[str,Array]]
            {hist name: {stat name: stat val, ...},
             ...}
            For consistency, statistics should follow the names used in `scipy.stats`:
            - 'mean'
            - 'std'
        """
        raise NotImplementedError


    @classmethod
    def stationary_dist(cls, params: ModelParams) -> pm.Model:
        """
        Return a PyMC3 model corresponding to the process' stationary distribution
        with the current model parameters.
        """
        # Variable names must match those of the histories
        raise NotImplementedError

    # NB: In rare cases, stationary distributions methods may depend on the model
    #     (spec. GWN depends on dt), so we can't force them to be @classmethods
    def stationary_stats(
        self,
        params: Union[None, ModelParams, IndexableNamespace, dict]=None,
        _max_cost: int=10
        ) -> Union[ModelParams, IndexableNamespace]:
        """
        Public wrapper for _stationary_stats, which does type casting.
        Returns either symbolic or concrete values, depending on whether `params`
        is symbolic.

        .. Note:: If just one of the values of `params` is symbolic, _all_ of the
           returned statistics will be symbolic.

        Parameters
        -----------
        params: Any value which can be used to construct a `cls.Parameters`
            instance. If `None`, `self.params` is used.
        _max_cost: Default value should generally be fine. See
            `theano_shim.graph.eval` for details.
        """
        if params is None:
            params = self.params
        symbolic_params = shim.is_symbolic(params)
        # Casting into cls.Parameters is important because cls.Parameters
        # might use properties to define transforms of variables
        if not isinstance(params, self.Parameters):
            params = self.Parameters(params)
        stats = self._stationary_stats(params)
        if not symbolic_params:
            stats = shim.eval(stats)
        return stats

    def stationary_stats_eval(self):
        """
        Equivalent to `self.stationary_stats(self.params).eval()`, with the
        benefit that the compiled function is cached. This saves the overhead
        of multiple calls to `.eval()` if used multiple times.
        """
        # TODO: Attach the compiled function to the class ?
        #       Pro: -only ever compiled once
        #       Con: -would have to use placeholder vars instead of the model's vars
        #            -how often would we really have more than one model instance ?
        compile_key = 'stationary_stats'
        try:
            compiled_stat_fn = self._compiled_functions[compile_key]
        except KeyError:
            # NB: `shim.graph.compile(ins, outs)` works when `outs` is a dictionary,
            #     but only when it is FLAT, with STRING-ONLY keys
            #     So we merge the keys with a char combination unlikely to conflict
            flatstats_out = {"||".join((k,statnm)): statval
                             for k, statdict in self.stationary_stats(self.params).items()
                             for statnm, statval in statdict.items()}
            assert all(len(k.split("||"))==2 for k in flatstats_out)
                # Assert that there are no conflicts with the merge indicator
            compiled_stat_fn = shim.graph.compile([], flatstats_out)
            self._compiled_functions[compile_key] = compiled_stat_fn
        flatstats = compiled_stat_fn()
        # TODO: It seems like it should be possible to ravel a flattened dict more compactly
        stats = {}
        for key, statval in flatstats.items():
            key = key.split("||")
            try:
                stats[key[0]][key[1]] = statval
            except KeyError:
                stats[key[0]] = {key[1]: statval}
        return stats

# %% [markdown]
# ### Testing
#
# Subclasses may optionally define a `get_test_parameters` class method, which is essentially a hard-coded default prior. This is useful to run tests without needing to specify a prior.
#
# :::{margin} Code
# `Model`: Test parameters
# :::

    # %%
    # TODO: Mechanism to impose constraints (e.g. two submodels with same M)
    @add_to('Model')
    @classmethod
    def get_test_parameters(cls, rng: Union[None,int,RNGenerator]=None):
        """
        :param:rng: Any value accepted by `numpy.random.default_rng`.
        """
        rng = np.random.default_rng(rng)  # Use `rng` to draw random parameters
        raise NotImplementedError


# %% [markdown]
# (priors)=
# ## Priors
#
# _Priors_ are defined as [PyMC3](https://docs.pymc.io/) models. This allows us to
#
# - Define priors by simply naming distributions and setting their parameters.
#   PyMC3 combines all priors and constructions one log likelihood function for the whole prior.
# - Let PyMC3 use its default transformations to [_unbounded domains_](https://docs.pymc.io/pymc-examples/examples/pymc3_howto/api_quickstart.html#Automatic-transforms-of-bounded-RVs) for better numerical stability.
# - Define certain parameters as deriving from others using [`Deterministics`](https://docs.pymc.io/pymc-examples/examples/pymc3_howto/api_quickstart.html#Deterministic-transforms).
# - Make certain parameters [`Constant`](https://docs.pymc.io/api/distributions/discrete.html#pymc3.distributions.discrete.Constant).
#   This is a simple and effective way of masking a variable from optimization.[^1]
#
# Since priors generally depend on some parameters (like the dimension of input space), the objects in the `priors` collection are in fact *factory functions* (which return PyMC3 models).
# This also avoid creating a bunch of unneeded PyMC3 models just by importing the modules.
#
# [^1]: For more complex masks (e.g. only a subset of components in a connectivity matrix), one should be able to define the unmasked components as a lower-dimensional variable and then construct the full variable with concatenation and reshaping operations. However this has not been tested.
#
# :::{note}
# Since `Prior` instances are meant to represent model priors, they must not define [observed](https://docs.pymc.io/pymc-examples/examples/pymc3_howto/api_quickstart.html#Observed-Random-Variables) variables.
# :::
#
# :::{important}
# Prior variables are mapped model parameters based on their name. So if your model defines a parameter “alpha”, your prior _must_ define a variable named “alpha”. It can, however, by defined as constant or via a deterministic transformation of other variables.
# :::

# %% [markdown]
# (parameter-spaces)=
# ### Parameter spaces

# %% [markdown]
# (prior-types-and-spaces)=
# :::{sidebar} Relation to RV types
#
# Random variables (RVs) in a prior can be of four[^num-RVs] PyMC3 types: `Constant`, `FreeRV`, `TransformedRV` or `Deterministic`, with each `TransformedRV` is associated to a `FreeRV` via a bijection.
#
# Optimization space
# ~ contains instances of `FreeRV`.
# ~ Represented by `~sinnfull.optim.base.OptimParams`.
#
# Prior space
# ~ contains instances of `FreeRV`, `Constant` and `TransformedRV`.
#
# Model space
# ~ contains instances of `Constant`, `Constant`, `FreeRV`, `TransformedRV` and `Deterministic`.
# ~ Represented by `~sinn.models.ModelParams`.
# :::
#
# [^num-RVs]: PyMC3 also defines the `ObservedRV` type, but by definition a prior should not contain observed variables.

# %% [markdown]
# Note that we have
#
# - **Random variables** (RVs) which are constructed with one of the PyMC3 distributions.
#   These form what we call the _prior space_.
# - **Bounded RVs**, which are associated to unbounded RVs via bijections.
#   These are automatically defined by PyMC3.
#   Since the unbounded variables are those with respect to which optimizers will differentiate,
#   we call the space of unbounded variables the _optimization space_.
# - RVs which are the result of **determistically transforming** other RVs.
#   These transformations are used to define parameters of the model which are derived from others.
#   We call the space of variables defined by the model the _model space_
#
# There are therefore two types of transformations:
#
# - _Bijective maps_ from _prior space_ to _optimization space_.
# - _Surjective maps_ from _prior space_ to _model space_.
#
# [![Transformations between parameter spaces](parameter_spaces.svg)](https://mermaid-js.github.io/mermaid-live-editor/#/edit/eyJjb2RlIjoiZ3JhcGggUkxcbiAgICBBW29wdGltIHNwYWNlXSAtLT58YmFja3dhcmR8QltwcmlvciBzcGFjZV1cbiAgICBCIC0tPnxmb3J3YXJkfEFcbiAgICBCIC0tPnxEZXRlcm1pbmlzdGljc3xDW21vZGVsIHNwYWNlXSIsIm1lcm1haWQiOnsidGhlbWUiOiJkZWZhdWx0In0sInVwZGF0ZUVkaXRvciI6ZmFsc2V9)

# %% [markdown]
# :::{admonition} More on transformed variables
# :class: dropdown
#
# By default, PyMC3 transforms all variables such that they have unbounded support on $\mathbb{R}$. For example, if a model includes `W = pm.Halfnormal('W', ...)`, than a private random variable `W_log__`, is created, using $\log$ to map $(0, \infty)$ to $(-\infty, \infty)$. This is generally desirable for numerical stability, but it is also possible to prevent the creation of a transformed variable by passing `transform=None` when creating the random variable.
#
# If a model parameter is specified in terms of a transformed random variable, this is best done by defining an `~pm.Deterministic` variable. For example, the code below defines $a$ as the row sums of a Gaussian matrix:
#
# ```python
# with Prior() as prior:
#     _a_mat = pm.Normal('_a_mat', mu=-1, sigma=1, shape(3, 3))
#     a      = pm.Deterministic(_a_mat.sum(axis=1))
# ```
#
# Note the use of a leading underscore in `_a_mat`: this indicates to `Prior` that this is an intermediate variable which is not part of the model. As such, it will not be included in the dictionary returned by `~Prior.random`.
#
# The disadvantage of `~pm.Deterministic` variables is that they don't provide a _log det jacobian_ and thus are not invertible. In general this should not be required, but if it is, it might be possible to specify the transformation using a `~pm.transforms.TransformedVar` instead. Since this is only lightly documented within PyMC3's developer documentation, we've compiled some WIP notes in the [developer docs](../../docs/Understanding_PyMC3_Transforms.ipynb). Note that we've never actually used a custom transformation in our models, and can't speak to their ultimate usefulness for this purpose.
#
# :::

# %% [markdown]
# :::{margin} Code
# `Prior`: Space transformations
# `Prior`: Variable substitution
# `Prior`: Sampling
# :::

# %% tags=["hide-input"]
class Prior(PyMC_Model):
    """
                                           forward
                                          --------->
    model_vars                 prior_vars            optim_vars
               <--------------            <---------
                Deterministics             backward
    """

    # Allow tag filters to be overspecified
    def __getattr__(self, attr):
        if hasattr(PyMC_Model, '__getattr__'):
            try:
                return super().__getattr__(attr)
            except AttributeError:
                pass
        if attr != '_tags' and attr in getattr(self, '_tags', set()):
            return self
        else:
            raise AttributeError(
                f"'{attr}' was not found in {type(self).__name__}.")

    # Replace '_' in prefix with '.', so it matches hierarchical ParameterSet
    # Since '.' is never used in var names, it has the advantage of allowing
    # to split the model name.
    @property
    def prefix(self):
        return f"{self.name}." if self.name else ""

    def no_observed_RVs(self):
        if self.observed_RVs:
            raise RuntimeError("A prior should not contain any observed RVs.\n"
                               f"Observed random variables: {self.observed_RVs}")

    @property
    def optim_vars(self) -> Dict[str,pm.model.FreeRV]:
        """
        Return non-constant free RVs as a dictionary of {name:var} pairs.

        Returns
        -------
        dict, sorted by name
        """
        self.no_observed_RVs()
        var_dict = {θ.name: θ for θ in self.free_RVs
                    if not isinstance(θ.distribution, pm.Constant)}
        return {θnm: var_dict[θnm] for θnm in sorted(var_dict)}

    @property
    def model_vars(self) -> Dict[str,pm.model.PyMC3Variable]:
        """
        Take the set of all named vars, and remove those corresponding
        to transformed or intermediate variables.
        These are respectively identified by checking whether they
        end with "__" or start with "_".

        Returns
        -------
        dict, sorted by name
        """
        self.no_observed_RVs()
        var_dict = {}
        for θname in sorted(self.named_vars.keys()):
            if not (pm.util.is_transformed_name(θname)
                    or θname.split('.')[-1].startswith("_")):
                # Starting with _ is our own convention for intermediate vars
                var_dict[θname] = self.named_vars[θname]
        return var_dict

    @property
    def _pymc_model_vars(self) -> Dict[str,pm.model.PyMC3Variable]:
        """
        Return the set of intermediate variables – those that are either
        public free RVs, or are related by a bijection to a hidden (optim)
        free RVs.
        In practice, this is the model_vars, before the application of
        deterministic transforms.

        Returns
        -------
        dict, sorted by name
        """
        var_dict = {}
        for θ in self.unobserved_RVs:
            if isinstance(θ, pm.model.TransformedRV):
                # Include transformed vars instead of their associated optim var
                var_dict[θ.name] = θ
            elif isinstance(getattr(θ,'distribution',None), pm.transforms.TransformedDistribution):
                # Don't include optim vars associated to a transformed var
                pass
            elif isinstance(θ, pm.model.FreeRV):
                # Include optim vars which are not associated to a transformed var
                var_dict[θ.name] = θ
            else:
                # Don't include deterministics
                pass
        # Sort by var name
        return {nm: var_dict[nm] for nm in sorted(var_dict)}

    def forward_transform(self, θval: Union[Number,Array], θname: str,
                          return_transformed_name: bool=True):
        """
        Transform a parameter value from a pymc model to optimization space.
        No-op if parameter is not transformed.
        (Pymc model space is intermediate between model and optim spaces.
        We can't start forward-transform from model space because deterministics
        are not bijective.)
        Raises an error if parameter is not a model parameter.
        :param:θval: The value to transform.
        :param:θname: Name of the parameter it corresponds to.
        :param:return_transformed_name: (Optional) Whether to also return
            the name of the transformed variable.

        Returns
        -------
        if `return_transformed_name` == True:
            θname, θval
        otherwise:
            θval
        """
        θ = self._pymc_model_vars[θname]  # By restricting to model_vars, we will catch
                                          # errors where we attempt to transform a
                                          # non-model variable
        if isinstance(θ, pm.model.TransformedRV):
            θname = θ.transformed.name
            θval = θ.distribution.transform.forward_val(θval)
        if return_transformed_name:
            return θname, θval
        else:
            return θval

    def forward_transform_params(self, Θ: Union[IndexableNamespace,Dict]):
        """
        Convenience function for transforming all parameters in a set.
        Applies `self.forward_transform` for each param in Θ.
        At present the values of `Θ` must not be shared or symbolic.

        Returns
        -------
        IndexableNamespace
            if Θ is an IndexableNamespace
        dict
            if Θ is a dict
        """
        # FIXME: `model_vars` is computed on each call to `forward_transform`
        if isinstance(Θ, (ModelParams,IndexableNamespace)):
            transformed_vars = [self.forward_transform(θ, θname) for θname, θ in Θ]
            return IndexableNamespace(
                **{θname: θval for θname, θval in transformed_vars})
        elif isinstance(Θ, dict):
            transformed_vars = [self.forward_transform(θ, θname) for θname, θ in Θ.items()]
            return {θname: θval for θname, θval in transformed_vars}
        else:
            raise TypeError("`forward_transform` expect a ModelParams argument. "
                            f"Received: {Θ} (type {type(θ)}).")


    # PyMC3 transforms don't provide a `backward_val`, so we need to compile
    # the theano function. We cache this since it is expension.
    @property
    @functools.lru_cache
    def _backward_transform(self):
        # Some of the model_vars may be constants, in which case they don't
        # appear in optim_vars => we need to replace them ourselves with their value
        ovars = list(self.optim_vars.values())
        mvars = list(self.model_vars.values())
        givens = {}
        for v in mvars:
            if isinstance(getattr(v, 'distribution', None), pm.Constant):
                givens[v] = shim.constant(v.tag.test_value, dtype=v.dtype)
        f = theano.function(ovars, mvars, givens=givens)
            # We use `theano` since with PyMC3, we will always have Theano
            # objects. If we used `shim.graph.compile`, it would complain
            # that Theano isn't loaded (within shim).
        # mvars2 = [shim.graph.clone(v, replace=givens) for v in mvars]
        # f = shim.graph.compile(ovars, mvars2, on_unused_input='ignore')
        return f

    def backward_transform_params(self, Θ: Union[dict,IndexableNamespace]) -> IndexableNamespace:
        """
        Transform all parameters in a set from optimization to model space.
        """
        if isinstance(Θ, IndexableNamespace):
            Θ = Θ.__dict__
        return {θnm: θ for θnm, θ in zip(self.model_vars,
                                         self._backward_transform(**Θ))}

    def sub_optim_vars(self, obj, optim_vars: ModelParams):
        """
        Do two things:
        - Replace “model space” (transformed) by “optimization space” (untransformed)
          variables.
        - Replace prior variables by the set of variables used by the optimizer.
          (This set living in optimization space, we need to do the first
          substitution first.)
        """
        self.no_observed_RVs()
        if isinstance(obj, (list,tuple,set,frozenset)):
            return type(obj)(self.sub_optim_vars(θ) for θ in obj)
        elif isinstance(obj, dict):
            return {k: self.sub_optim_vars(θ) for k,θ in obj.items()}
        elif isinstance(obj, (ModelParams,IndexableNamespace)):
            return IndexableNamespace(
                **{θname: self.sub_optim_vars(θ) for θname, θ in obj})
        elif isinstance(obj, shim.config.GraphTypes):
            symb_inputs = {i.name: i for i in shim.graph.symbolic_inputs(obj)
                           if hasattr(i, 'name')}
                # NB: symb_inputs will also include history data & time index vars.
            # First replace model variables by their transformed versions
            model_vars = self.model_vars  # Recall: model_vars = transformed(optim_vars)
            forward_replace_dict = {symb_inputs[nm]: model_vars[nm]
                                    for nm in model_vars if nm in symb_inputs}
            new_expr = shim.graph.clone(obj, replace=forward_replace_dict)
            # Now replace the transformed vars by the optimization vars of same name
            # Also replace constants RVs by their value
            optim_replace_dict = {self.optim_vars[nm]: optim_var
                                  for nm, optim_var in optim_vars}
            for θ in self.unobserved_RVs:
                if isinstance(getattr(θ,'distribution', None), pm.Constant):
                    optim_replace_dict[θ] = θ.tag.test_value  # For contants, test value = value
            new_expr = shim.graph.clone(new_expr, replace=optim_replace_dict)
            # At this point there should no variables from the prior left
            # in the graph – they should all be replaced by the vars the
            # optimizer will optimize.
            prior_vars_in_graph = \
              set(shim.graph.symbolic_inputs(new_expr)) & set(self.unobserved_RVs)
            if prior_vars_in_graph:
                raise RuntimeError("The following variables from the prior are "
                                   "still in the expression graph: "
                                   f": {prior_vars_in_graph}.\nThey should "
                                   "by replaced by optimization variables.")
            return new_expr
        else:
            raise TypeError("`sub_optim_vars` expects either a single symbolic "
                            "expression or a container of symbolic expressions "
                            f"(Sequence, dict, ModelParams).\nReceived: {obj}.")

    # # TODO? At the moment we don't need this, but it's a natural analog to
    # #       sub_optim_vars and should be very similar
    # def sub_model_vars(self, obj, model):
    #     if isinstance(Θ, (list,tuple,set,frozenset)):
    #         return type(Θ)(self.sub_model_vars(θ) for θ in Θ)
    #     elif isinstance(Θ, dict):
    #         return {k: self.sub_model_vars(θ) for k,θ in Θ.items()}
    #     elif isinstance(Θ, (ModelParams,IndexableNamespace)):
    #         return IndexableNamespace(
    #             **{θname: self.sub_model_vars(θ) for θname, θ in Θ})
    #     elif isinstance(Θ, shim.config.GraphTypes):
    #         # TODO: backward_replace_dict
    #         new_expr = shim.clone(Θ, replace=backward_replace_dict)
    #         non_model_vars_in_graph = \
    #             set(shim.graph.symbolic_inputs(new_expr)) - set(self.model_vars)
    #         if non_model_vars_in_graph:
    #             raise RuntimeError("The following symbolic inputs are still in "
    #                                "the expression graph but not optimization "
    #                                f"variables: {non_model_vars_in_graph}")
    #         return new_expr
    #     else:
    #         raise TypeError("`sub_model_vars` expects either a single symbolic "
    #                         "expression or a container of symbolic expressions "
    #                         f"(Sequence, dict, ModelParams).\nReceived: {obj}.")

    def random(self, key: Tuple[int,...], space: str='model') -> Dict[str,Array]:
        """
        Wrapper around `sample_prior_predictive` which does the following:
        - Converts `key` to an integer seed (which PyMC3 still relies on)
        - Draws only one sample
        - Removes the sample dimension from each variable
        - Returns variables either only in the model or optimization spaces

        Parameters
        ----------
        key: Key provided to `~sinnfull.rng.get_seedsequence` to generate
             a random seed.
        space: 'model' (default) | 'optim'
            Whether to return model or optimization parameters.
            The mapping from optimization to model space is always injective.
            However, if the prior involves deterministic variables, it may
            not be bijective.
        """
        if space == 'optimization':
            space = 'optim'
        self.no_observed_RVs()
        seed = get_seedsequence(key, exists_ok=True).generate_state(1)[0]
            # exists_ok=True is safe if `key` is not used elsewhere in the code
        if space == 'model':
            var_names = sorted(self.model_vars)
        elif space == 'optim':
            # The problem with optim vars: PyMC3 will silently drop transformed
            # variables. So instead we draw from the set of “pymc model vars”,
            # which intermediate between the model and optim space.
            # After drawing, we will transform them to the optim space.
            # FIXME: The isinstance(..., Constant) is copied from optim_vars()
            model_vars = self._pymc_model_vars
            var_names = [θnm for θnm in sorted(self._pymc_model_vars)
                         if not isinstance(model_vars[θnm].distribution, pm.Constant)]
        else:
            raise ValueError("`var_names` must be either 'model' or 'optim'. "
                             f"Received '{space}'.")
        Θ = pm.sample_prior_predictive(
            samples=1,
            random_seed=seed,
            var_names=var_names,
            model=self)
        # Each sampled var will be wrapped in a list of length 1 (since we
        # asked for one sample). However constant are not returned wrapped in a
        # list, so we need to treat the two separately.
        all_rv = {rv.name: rv for rv in
                  itertools.chain(self.unobserved_RVs, self.observed_RVs)}
            # NB: all_rv[θname] instead of getattr(self, θname) works even when names include '.'
        constant_names = {θname for θname in Θ
                          if not shim.graph.symbolic_inputs(all_rv[θname])}
        constants = {θname: θ for θname, θ in Θ.items()
                     if θname in constant_names}
        samples = {θname: θ for θname, θ in Θ.items()
                   if θname not in constant_names}
        assert all(len(θvalue)==1 for θvalue in samples.values()), \
            "`sample_prior_predictive` did not return exactly 1 sample per parameter."
        samples = {θname: θvalue[0] for θname, θvalue in samples.items()}
        # Merge constants & samples back together and re-sort
        Θ_dict = {**constants, **samples}
        Θ_dict = {θname: Θ_dict[θname] for θname in sorted(Θ)}
        if space == 'optim':
            Θ_dict = self.forward_transform_params(Θ_dict)
        return Θ_dict

    def logpt_replace(self, replace: Dict[str, shim.config.SymbolicType],
                      assert_replaced_all: bool=True):
        """
        Return logpt, applying the provided replacement dictionary.
        This correctly substitutes transformed variables, meaning that if a
        variable name in `replace` matches a transformed variable, the
        associated value is first forward-transformed, that substituted
        for the corresponding transformed variable.

        Parameters
        ----------
        replace: Dict of {var name: new variable} pairs.
            Model variables matching `var name` are replaced by `new variable`
        assert_replaced_all: If true (default), perform an assert that all model
            variables were replaced before returning the new computational graph.
        """
        rv_list = self.unobserved_RVs.copy()
        for rv in rv_list[:]:
            if hasattr(rv, 'transformed'):
                rv_list.remove(rv.transformed)
        replace_dict = {}
        for rv in rv_list:
            if rv.name in replace:
                replace_var = replace[rv.name]
                if isinstance(replace_var, (Number, np.ndarray)):
                    # If we don't explicitely make the new value a Constant,
                    # numeric data types are converted to shared by `clone`
                    replace_var = shim.constant(replace_var, name=rv.name, dtype=rv.dtype)
                if hasattr(rv, 'transformed'):
                    replace_dict[rv.transformed] = rv.transformation.forward(replace_var)
                else:
                    replace_dict[rv] = replace_var

        logpt = shim.graph.clone(self.logpt, replace=replace_dict)
        if assert_replaced_all:
            # Assert all PyMC3 variables were replaced by model variables
            assert all(v not in rv_list for v in shim.graph.symbolic_inputs(logpt))
        return logpt


# %% [markdown] tags=["remove-cell"]
# PROBLEM: Serializing PureFunction includes the decorator(s). However,
#   we don't want to include the @ObjectiveFunction decorator (tags are
#   already stored, and could differ from those in the decorator?)
# SOLUTION/HACK: Create a new PureFunction type, which wraps the json_encoder
#   for PureFunction with another which removes lines starting with '@ObjectiveFunction'.
# AND YET: Now that tags are stored with `set` instead of `frozenset`, adding
#   tags should be harmless (unless they are dynamically removed).
#   Keeping the hack for now since it works and I don't care to work out
#   the magic done by ObjectiveFunction.\_\_new\_\_.

# %% [markdown]
# (objective-functions)=
# ## Objective functions
#
# Mathematically, an objective function $l$ is function from the combined parameter ($θ$) and latent ($η$) spaces, to a scalar: $l(θ,η) \to \mathbb{R}$. In our framework, both $θ$ and $η$ can be retrieved from the model, so in code we have `l(model,...) -> float`.
#
# We consider two types of objective functions:
#
# - _Accumulated_ objective functions, which additionally take a time point $t$: `l(model, t)`.
#   These are meant to be evaluated at multiple time points, and their results summed (accumulated).
# - _Regularizers_, which take no extra arguments: `l(model)`.
#   For parameters, this is functionally equivalent to defining `Prior`.
#
# Since we don't currently implement regularizers, they are not discussed further.
#
# :::{note}
# Mathematically, one expects an objective function to be [pure](https://en.wikipedia.org/wiki/Pure_function), and this is required for the proper functionning of the framework. See _Serializability_ below.
# :::
#
# :::{note}
# The [`Optimizer`](sinnfull/optim/base) class names its objective function “*logp*”, which is borrowed from the name of the analogous function in [PyMC3](https://docs.pymc.io/Probability_Distributions.html). In fact any scalar objective function is acceptable; it doesn't have to be a likelihood, or even a proper probability. However, if it does derive from a probability, it should correspond to the _log_ probability, such that it is compatible with an accumulated objective.
# :::
#
# :::{note}
# Objective functions are attached to the model being optimized, so their `self` arguments resolves to this model (note that the first argument *must* be named self). The second argument to the objective function must be a integer time index, corresponding to a time point on the model's *TimeAxis*.
# :::
#
# Beyond semantically identifying a function as being an objective, the `ObjectiveFunction` class adds the following functionality:
#
# - **Serializability** Since objective functions are [pure](https://en.wikipedia.org/wiki/Pure_function), they can be serialized. This requires them to use only variables in their own scope (i.e., their arguments `self` and `k`) – in particular, they cannot use any external packages imported by the modules.
#   Exceptions are the *numpy* (`np`), *math* (`math`) and *theano_shim* (`shim`), which are specially provided by the deserializer.
# - **Basic arithmetic** Objective functions support basic arithmetic operations as *functions*.
#
# The support for arithmetic allows one to construct objectives from more primitive ones, e.g.
#
# ```python
# objective = γ*objective1 + (1-γ)*objective2
# ```
#
# would define a new function `objective` callable as `objective(model, k)`. This feature helps reduce the proliferation of objective definitions.

# %% [markdown]
# ### Semantic tags
#
# Although any string can be used as a tag, objective functions currently ascribed additional meaning to specific strings. This is used to inform users on the intended use of a function as well provide validation – for example, preventing both `forward` and `backward` tags from being used simultaneously.
#
# :::{admonition} Partly historical note
# :class: dropdown
#
# The list below describes the tags that were at some point defined, and which are still recognized by `ObjectiveFunction`. Since their expected usefulness has faded compared to previous iterations of the framework, their future is uncertain.
#
# - `forward`: Objective function corresponding to forward simulation. Appropriate for fitting parameters and latent variables; when fitting latents, the dependence on the simulation model makes this objective less reliable at the edges of the time trace.
#
# - `backward`: Objective function corresponding to backward simulation; not currently supported (`SGDOptimizer` would need a second, backward-integrated model)
#
# - `nodyn`: Purely pointwise objective functions; considers only whether the latents are consistent with observations. Not appropriate for fitting parameters. Fitting latents with a `nodyn` objective function leads to discontinuous latents which may not be consistent with the model, but avoids interplay between parameters and latents, which can lead to instabilities.
#
# - `global`: An objective function which doesn't take time points at all, but the model. Useful for specifying priors.
#
# - `regularizer`: A synonym for `global`.
#
# - `prior`: Functionally equivalent to `regularizer`, with the additional semantic meaning of “prior”.
#
# - `se`: “Squared-error”. Consistency with observations is computed by the squared difference between observations and predictions. This can be used as an ad-hoc substitute for an observation model.
#   This tag does not imply that parameter errors are computed by squared-error; it relates only to the dynamics (the latent error).
#
# - `l1`: $L^1$ loss. Consistency with observations is computed by the absolute difference between observations and predictions. This can be used as an ad-hoc substitute for an observation model.
#   This tag does not imply that parameter errors are computed with $L^1$; it relates only to the dynamics (the latent error).
#
# - `log L`: The objective corresponds to a log-likelihood, although it need not be normalized. In contrast to `se`, this is used to indicate that _all_ error terms stem from a likelihood, and therefore that the objective function as a whole may be interpreted as the unnormalized joint log probability $\log p(x | \Theta) + C$.
#
# These attributes can be specified as tags. Tags may be used by functions to decide how to use an objective function, or just to validate that it is of an expected type. They are also useful metadata for users, and we may in the future use them to automatically construct the API.
#
# The following sets of tags are mutually exclusive, for hopefully obvious reasons:
# - `forward`, `backward`, `nodyn`, `global`
# - `log L`, `se`
#
# :::

# %% [markdown]
# ### Definition - ObjectiveFunction

# %% tags=["hide-input"]
class PureFunctionObjective(PureFunction):
    submodel : str=""
    @classmethod
    def validate(cls, value):  # Without this wrapper, values would be cast to PureFunction
        if isinstance(value, PureFunctionObjective):
            return value
        elif isinstance(value, Callable):
            return PureFunctionObjective(value)
        else:
            return super().validate(value)
    def __call__(self, model, *args, **kwargs):
        if self.submodel:
            model = getattr(model, self.submodel)
        return super().__call__(model, *args, **kwargs)

class CompositePureFunctionObjective(CompositePureFunction, PureFunctionObjective):
    @classmethod
    def validate(cls, value):
        if isinstance(value, PureFunctionObjective):
            return value
        elif isinstance(value, Callable):
            return CompositePureFunctionObjective(value)
        else:
            return super().validate(value)
def pure_function_encoder_wrapper(func):
    s = PureFunction.json_encoder(func)
    if isinstance(s, str):
        # CompositePureFunction returns tuple; nothing to do in that case
        s = '\n'.join(line for line in s.split('\n')
                      if not line.startswith('@ObjectiveFunction'))
    return s
mtbtyping.add_json_encoder(PureFunctionObjective, pure_function_encoder_wrapper,
                           priority=1)
# HACK !!!!!! We reorder the json_encoders while preserving refs to the variable
import sinnfull
for k in list(sinnfull.json_encoders.keys()):
    del sinnfull.json_encoders[k]
sinnfull.json_encoders.update(mtbtyping.json_encoders)

# %% [markdown]
# :::{margin} Code
# `ObjectiveFunction`: Function artithmetic
# `ObjectiveFunction`: Tag validation
# :::

# %% tags=["remove-cell"]
from pydantic.main import ModelMetaclass as BaseModelMeta
class ObjectiveFunctionMeta(BaseModelMeta):
    def __getattr__(cls, attr):
        if hasattr(BaseModel, '__getattr__'):
            try:
                return BaseModel.__getattr__(cls, attr)
            except AttributeError:
                pass
        if attr != 'tags' and attr in cls.tags:
            return cls
        else:
            raise AttributeError(f"{cls} does "
                                 f"not define the attribute '{attr}'.")



# %% tags=["hide-input"]
class ObjectiveFunction(BaseModel, abc.ABC, metaclass=ObjectiveFunctionMeta):
    """
    A wrapper around objective functions, allowing serialization and
    basic arithmetic operations (provided by smttask.PureFunction).
    Allows additional metadata to be attached in the form of tags, which
    functions like `~sinnfull.optim.SGDOptimizer` may use to determine
    how to use the function, or simply validate that it is of an expected type.

    Use as a decorator, or instantiate with a function argument.
    `ObjectiveFunction` is an abstract base class; during instantiation,
    a subclass is returned: either an `AccumulatedObjectiveFunction` or
    `Regularizer`, which have different expected signatures.
    The wrapped function must match the expected signature.

    Exposes a `submodel` attribute. This is used with composite model, to
    indicate which submodel should be passed as "model" argument to the
    objective function.

    TODO: Validate that the wrapped function has the expected signature.

    >>> from sinnfull.utils import ObjectiveFunction
    >>>
    >>> @ObjectiveFunction
    >>> def f(model, k):
    >>>     ...
    >>> def g(model, k):
    >>>     ...
    >>> g_obj = ObjectiveFunction(g)
    >>>
    >>> γ = 0.2
    >>> h = γ*f + (1-γ)*g_obj
    >>> # Use as
    >>> h(model, tidx)
    """
    func: PureFunctionObjective
    tags: Set[str]=set()

    # Attempting to use a tag listed in `disallowed_tags` will raise an error.
    disallowed_tags: ClassVar[set] = set()
    # TODO: Building a smoother from forward + backward should be allowed.
    #       So should be adding an 'se' to an 'log L' (its just no longer a 'log L' cost)
    #       What do we want to prevent with excluded tags ?
    #       => We need a working optimizer which actually uses a backward
    #       objective to see how/if such exclusions should be used.
    exclusive_tags: ClassVar[List[set]] = [
        # {'forward', 'global'},
        # {'backward', 'global'},
        # {'se', 'l1', 'log L'}
        ]
    default_tags: ClassVar[set] = set()
        # Subclasses can define default tags, which are always added unless
        # they clash with an exclusive tag
    # Especially at this point, where I'm still figuring which tags are useful,
    # I find it useful to define synonyms so that the set of tags can evolve.
    # Format: {ref_tag: [synonym, …]}  : All synonyms are replaced by `ref_tag`
    synonym_tags: ClassVar[Dict[str, List[str]]] = {
        'regularizer': {'global'},
        'se': {'squared-error'} }

    ## Abstract methods – subclasses must implement these ##
    # Explanation: sinn.Model.accumulate inspects the signature of objective
    # functions, and it must not contain variadic elements
    @abc.abstractmethod
    def __call__(self, *args):
        # Should select submodel if `self.submodel` is not None
        return self.func(*args)

    ## Attributes
    @property
    def __name__(self):  # Required by sinn.Model.accumulate
        return self.func.__name__

    ## Initialization ##
    # Two possible initializations. One can specify either
    #  - `func` -> Normal initialization (bare decorator)
    #  - `tag`  -> decorator with argument; return a new decorator
    #  - `func`, `tags` -> Non-standard, but used in arithmetic methods to create new instances
    # Note also that subclasses may define extra attributes, which are caught by **kws
    def __new__(cls, func=None, *, tags=None, **kws):
        if func is None and tags is None and len(kws) == 0:
            # Must allow empty args, b/c Pydantic uses that in copy()
            return super().__new__(cls)
            #     raise TypeError("At least one of `func`, `tags` must be specified.")
        elif isinstance(func, set) and tags is None:
            # Allow `tags` to be specified without keyword
            tags, func = func, tags
        if func is None and len(kws) == 0:
            # If only tags are specified, it means the usage was @ObjectiveFunction(tags={})
            # => Return a decorator function
            tags = cls.validate_tags(tags)
            if 'regularizer' in tags:
                subcls = Regularizer
            else:
                subcls = AccumulatedObjectiveFunction
            def decorator(func):
                return subcls(func=func, tags=tags)
            return decorator  # Not an ObjectiveFunction => won't trigger __init__
        else:
            return super().__new__(cls)

    # Report and set the "submodel" attribute as the one of 'func'
    # ('submodel' must be attached to 'func' in order to not be lost during
    #  function arithmetic)
    def __setattr__(self, attr, value):
        if attr == "submodel":
            self.func.submodel = value
        else:
            super().__setattr__(attr, value)
    # Allow tag filters to be overspecified
    def __getattr__(self, attr):
        if attr == "submodel":
            return self.func.submodel
        if hasattr(BaseModel, '__getattr__'):
            try:
                return super().__getattr__(attr)
            except AttributeError:
                pass
        if attr in self.tags:
            return self
        else:
            raise AttributeError(
                f"'{attr}' was not found in {type(self).__name__}.")

    ## Validation ##
    ## It would be convenient to be able to validate serialized strings as well,
    ## but I get "unexpected keyword argument 'value'" when I do this:
    # @classmethod
    # def validate(cls, *args, **kwargs):  # Allow deserializing strings
    #     if len(args) == 1 and isinstance(args[0], str):
    #         return cls.parse_raw(args[0])
    #     else:
    #         super().validate(*args, **kwargs)

    @validator('func')
    def parse_func_from_str(cls, func):
        if isinstance(func, str):
            func = mtbserialize.deserialize_function(func)
        return func

    @validator('tags')
    def validate_tags(cls, tags):
        ## Validate against disallowed_tags
        if not tags:
            return set()
        if tags & cls.disallowed_tags:
            raise ValueError(f"The following tags are disallowed for {cls}: "
                             f"{tags & cls.disallowed_tags}")
        for excl_combo in cls.exclusive_tags:
            if len(tags & excl_combo) > 1:
                raise ValueError("The following tags are mutually exclusive; "
                                 "you may only specify one: "
                                 f"{tags & excl_combo}.\n")
        ## Add default tags, if they don't clash with an existing tag
        for def_tag in cls.default_tags:
            if def_tag not in tags:
                for excl_combo in cls.exclusive_tags:
                    if len(tags & excl_combo) == 0:
                        tags.add(def_tag)
        ## Normalize tags by replacing synonymous tags
        # First invert the dictionary
        syn_subs = {syn: ref_tag
                    for ref_tag, syn_list in cls.synonym_tags.items()
                    for syn in syn_list}
        assert len(syn_subs) == sum(len(syn_list) for syn_list in cls.synonym_tags.values())
            # Ensure there are no conflicts between synonym mappings
        # Now performs substitutions until tag list no longer changes
        # (we allow for a ref tag to itself be mapped to another ref tag)
        normalized_tags = {}
        while normalized_tags != tags:
            normalized_tags = {syn_subs.get(tag, tag) for tag in tags}
        tags = normalized_tags

        return set(tags)


    ## Function arithmetic ##
    # REMARK: sinn.Model.accumulate inspects the signature of the composite
    #   objective function, and it must not contain variadic elements (and at the
    #   same time check that signatures match) ? Could we do this programmatically
    #   in PureFunction, i.e. assign to __call__ the signature of self.func ? If
    #   removed, also remove `import operator`
    # BEHAVIOUR ON TAGS: Tag sets are merged, and an error is raised if the
    #   result has more than one mutually exclusive tags.
    def validate_op_args(self, other):
        """
        Return the arguments on which the operation should be applied.
        Raises ValueError if `other` is neither a plain data type nor an ObjectiveFunction.
        """
        if isinstance(other, Callable) and not isinstance(other, ObjectiveFunction):
            raise ValueError("Operations on ObjectiveFunctions are only permitted "
                             "with other ObjectiveFunctions or plain data types")
        b = other.func if isinstance(other, ObjectiveFunction) else other
        return self.func, b
    def combine_tags(self, other):
        # if isinstance(other, ObjectiveFunction) and self.tags != other.tags:
        #     raise ValueError("Operations between ObjectiveFunctions are only "
        #                      "permitted if both have the same tags.\n"
        #                      f"Arg 1 tags: {self.tags}\n"
        #                      f"Arg 2 tags: {other.tags}")
        if isinstance(other, ObjectiveFunction):
            new_tags = self.tags | other.tags
            # Validate both ways, to make sure it doesn't make a difference.
            new_tags1 = self.validate_tags(new_tags)
            new_tags1 = other.validate_tags(new_tags1)
            new_tags2 = other.validate_tags(new_tags)
            new_tags2 = self.validate_tags(new_tags2)
            assert new_tags1 == new_tags2
            return new_tags1
        else:
            return self.tags

    def __abs__(self):
        return type(self)(func=CompositePureFunctionObjective(operator.abs, self.func), tags=self.tags)
    def __neg__(self):
        return type(self)(func=CompositePureFunctionObjective(operator.neg, self.func), tags=self.tags)
    def __pos__(self):
        return type(self)(func=CompositePureFunctionObjective(operator.pos, self.func), tags=self.tags)
    def __add__(self, other):
        a, b = self.validate_op_args(other)
        tags = self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.add, a, b), tags=tags)
    def __radd__(self, other):
        a, b = self.validate_op_args(other)
        tags = self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.add, b, a), tags=tags)
    def __sub__(self, other):
        a, b = self.validate_op_args(other)
        tags = self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.sub, a, b), tags=tags)
    def __rsub__(self, other):
        a, b = self.validate_op_args(other)
        tags = self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.sub, b, a), tags=tags)
    def __mul__(self, other):
        a, b = self.validate_op_args(other)
        tags = self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.mul, a, b), tags=tags)
    def __rmul__(self, other):
        a, b = self.validate_op_args(other)
        tags = self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.mul, b, a), tags=tags)
    def __truediv__(self, other):
        a, b = self.validate_op_args(other)
        tags = self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.truediv, a, b), tags=tags)
    def __rtruediv__(self, other):
        a, b = self.validate_op_args(other)
        tags = combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.truediv, b, a), tags=tags)
    def __pow__(self, other):
        if isinstance(other, ObjectiveFunction):
            raise TypeError("Using ObjectiveFunctions as powers is not supported "
                            "because we do not have a valid use case for it.")
        assert self.tags == self.combine_tags(other)
        return type(self)(func=CompositePureFunctionObjective(operator.pow, self.func, other), tags=self.tags)

# %%
class AccumulatedObjectiveFunction(ObjectiveFunction):
    """
    A pointwise objective function, with the signature ``(model, k)``
    (``k`` being a time index).
    """
    func: PureFunctionObjective[[sinn.Model, int], float]

    disallowed_tags: ClassVar[set] = {'regularizer', 'global'}

    # Redefine __call__ so that it has the signature required by sinn.Model.accumulate
    def __call__(self, model, k):
        return self.func(model, k)

# %% tags=["remove-cell"]
# NB: We don't currently use regularizers, since priors are preferred for regularizing parameters,
#     so the code below is unmaintained.
#     There could be a use case however for regularizing latent variables.
class Regularizer(ObjectiveFunction):
    """
    A time-independent objective function, with the signature ``(model)``.
    Appropriate for an objective that doesn't depend on the model dynamics
    but only its parameters.
    A prior on the parameters falls in this category.
    """
    func           : PureFunctionObjective[[sinn.Model], float]
    prior          : Optional[Prior] = None
    disallowed_tags: ClassVar[set] = {'forward', 'backward', 'nodyn'}
    default_tags   : ClassVar[set] = {'regularizer'}

    ## Redefine __call__ so that it has the expected signature
    def __call__(self, model):
        return self.func(model)

    ## Validation
    @root_validator(pre=True)
    def parse_prior(cls, values):
        func = values.get('func', None)
        prior = values.get('prior', None)
        if (func is None) == (prior is None):
            raise TypeError("Exactly one of `func`, `prior` must be specified.")
        if isinstance(func, Prior):
            prior, func = func, prior
        if prior is not None:
            func = functools.partial(cls.sub_model_vars_into_logpt, prior=prior)
        values.update(func=func, prior=prior)
        return values

    @staticmethod
    def sub_model_vars_into_logpt(model: sinn.Model, prior: Prior):
        if not isinstance(prior, Prior):
            prior = Prior.validate(prior)
        subbed_logpt = prior.logpt_replace(model.params.dict())
        unsubbed_vars = [v for v in shim.graph.symbolic_inputs(subbed_logpt)
                         if v not in model.params.dict().values()]
        if unsubbed_vars:
            raise ValueError("The following symbolic variables in the prior "
                             f"were not substituted: {unsubbed_vars}.")
        return subbed_logpt

    ## Don't serialize 'func' if 'prior' is given
    def dict(self, *a, exclude=None, **kw):
        if self.prior:
            exclude = add_exclude_mask(exclude, {'func'})
        return super().dict(*a, exclude=exclude, **kw)
    def json(self, *a, exclude=None, **kw):
        if self.prior:
            exclude = add_exclude_mask(exclude, {'func'})
        return super().json(*a, exclude=exclude, **kw)

# %% tags=["remove-cell"]
ObjectiveFunction.update_forward_refs()
AccumulatedObjectiveFunction.update_forward_refs()
Regularizer.update_forward_refs()

# %% [markdown]
# ## Tagging

# %%

# `tag` is used to store tags in a '_tags' attribute
# It can be used as either a function or decorator

# NB: We need to support for both '_tags' and 'tags' attributes, since
#     Objectives use the latter. (The reason being that their tags are public
#     attributes)

tag = TagDecorator(TypeDict({
    Model: '_tags',
    Prior: '_tags',
    ObjectiveFunction: 'tags',
    ParameterSet: '_tags',
    }))
