# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metada_filter: -jupytext.text_representation.jupytext_version
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
# # Understanding PyMC3 Transforms

# %% [markdown]
# ## A simple example

# %%
import pymc3 as pm
from theano.printing import debugprint
import theano_shim as shim
import theano.tensor as tt
import matplotlib.pyplot as plt

# %% [markdown]
# We consider the following model

# %%
with pm.Model() as model:
    x = pm.Lognormal('x', mu=0, sigma=1)
    y = pm.Normal('y', mu=0, sigma=1)
model

# %% [markdown]
# This model has one transformed RV (`x`), and one non-transformed RV (`y`).

# %%
print(x, ":", type(x))
print(y, ":", type(y))

# %% [markdown]
# However the transformed variable `x` is just a convenience for the users; the “true” variable, which PyMC3 uses for sampling, is `x_log__`:

# %%
print("Free RVs:", model.free_RVs)
print(model.x_log__, ":", type(model.x_log__))

# %% [markdown]
# > **Important**: The log transform of $x$ has nothing to do with the fact that $x$ is a Lognormal. `Normal` and `Lognormal` are separate distributions, with their own implementations of `logp`:
# > ```python
# > def x.distribution.logp(value):
# >    return bound(
# >        -0.5 * tau * (tt.log(value) - mu) ** 2
# >        + 0.5 * tt.log(tau / (2.0 * np.pi))
# >        - tt.log(value),
# >        tau > 0,
# >)
# >
# > def y.distribution.logp(value):
# >     return bound((-tau * (value - mu) ** 2
# >                   + tt.log(tau / np.pi / 2.0)) / 2.0,
# >                  sigma > 0)
# >```
# >
# >Rather the transform has to do with mapping the domain of $x$, which is $(0, \infty)$, to $(-\infty, \infty)$. PyMC3 samples all variables in unbounded real space.

# %% [markdown]
# Each RV is tied to a stateless _distribution_:

# %%
print("x: ", x.distribution, ":", type(x.distribution))
print("y: ", y.distribution, ":", type(y.distribution))

# %% [markdown]
# In the case of `x`, its distribution has a non-`None` _transform_:

# %%
print("x.distribution.transform:", x.distribution.transform)
print("y.distribution.transform:", y.distribution.transform)

# %% [markdown]
# This is exactly the instance created with the _distributions.transforms_ module:

# %% [markdown]
# The “true” variable `x_log__` is tied to a _TransformedDistribution_, mapping the $(0, \infty)$ domain of `Lognormal` to $(-\infty, \infty)$. To this transformed distribution is attached the original log normal distribution:

# %%
x.distribution is x.transformed.distribution.dist

# %% [markdown]
# This we have the somewhat confusing relationship that a _transformed_ RV is associated to a _normal_ distribution, and a _normal_ (free) RV is associated to a _transformed_ distribution:
#
# ```
# TransformedRV  --------  FreeRV
#      |                     |
# Distribution ---- TransformedDistribution
# ```
# A set of attributes allows to move along these relationships (see [below](#Attribute-relationships)).

# %% [markdown]
# The methods `logp` and `random` are accessible as follows:
# - FreeRV
#   + logp
#   + random
# - Distribution
#   + logp
#   + random
# - TransformedRV
#   + ~logp~
#   + random
# - TransformedDistribution
#   + logp
#   + ~random~

# %% [markdown]
# ### Attribute relationships

# %%
import networkx as nx


# %%
class Node:
    def __init__(self, name, type):
        self.name = name
        self.type = type
    def __str__(self):
        return f"{self.name} ({self.type})"
    def __hash__(self):
        return hash(self.name)


# %%
G = nx.DiGraph()
labels = {}
edge_labels = {}


# %%
def add_node(a, label=None):
    G.add_node(a)
    if label is None:
        label = f"{str(a)} ({type(a).__name__})"
    labels[a] = label
def add_edge(start, stop, label):
    G.add_edge(start, stop)
    edge_labels[(start,stop)] = label


# %%
add_node(x)
add_node(x.transformed)
add_edge(x, x.transformed, ".transformed")
add_node(x.distribution)
add_edge(x, x.distribution, ".distribution")
add_node(x.distribution.transform)
add_edge(x.distribution, x.distribution.transform, ".transform")
add_node(x.transformed.distribution)
add_edge(x.transformed, x.transformed.distribution, ".distribution")
add_node(x.transformed.distribution.dist)
add_edge(x.transformed.distribution, x.transformed.distribution.dist, ".dist")
add_node(x.transformed.distribution.transform_used)
add_edge(x.transformed.distribution, x.transformed.distribution.transform_used, ".transform_used")

# %%
pos = {
    x: (0, 1),
    x.transformed: (1, .8),
    x.distribution: (0, 0),
    x.distribution.transform: (0, -1),
    x.transformed.distribution: (1, -.2),
}

# %%
#pos = nx.spring_layout(G)
nx.draw(G, pos=pos, labels=labels);
nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels);
plt.xlim(-0.7, 1.7)

# %% [markdown]
# ## Defining custom transforms

# %% [markdown]
# _pymc.distributions.transforms_ provides two types of transforms: _Transform_ and _ElemwiseTransform(Transform)_. The latter simply adds a default implementation of `jacobian_det`, using Theano to compute the gradient. This implementation is instructive in illustrating what shape the `jacobian_det` should return:
#
# ```python
# def ElemwiseTransform.jacobian_det(self, x):
#     grad = tt.reshape(gradient(tt.sum(self.backward(x)), [x]), x.shape)
#     return tt.log(tt.abs_(grad))
# ```
# - `sum` replaces the determinant (the Jacobian is diagonal) and is exchangeable with `gradient` in this case.
# - The result has the same shape as `x`, providing the log-det-abs-Jacobian for each output component separately.

# %%
import numpy as np
from pymc3.distributions.transforms import draw_values, floatX

class ScaleTransform(pm.transforms.Transform):
    name = "scale"
    
    def __init__(self, scaling):
        self.scaling = tt.as_tensor_variable(scaling)
    
    def backward(self, x):
        scaling = self.scaling
        return x / scaling
    
    def forward(self, x):
        scaling = self.scaling
        return x * scaling
    
    def forward_val(self, x, point=None):
        scaling = draw_values([self.scaling - 0.0], point=point)[0]
        return floatX(x * scaling)
    
    def jacobian_det(self, x):
        """
        Log of absolute value of Jacobian determinant of the backward transformation.
        
        Log(abs(det(J(x))))
        """
        grad = tt.reshape(1/self.scaling, x.shape)
        return tt.log(tt.abs_(grad))


# %%
with pm.Model() as model:
    z = pm.Normal('z', mu=np.zeros(4), sigma=np.ones(4), shape=(4,),
                  transform=ScaleTransform(np.array([1., 2., 3., 4.])))

# %% [markdown]
# We can check `jacobian_det` by comparing the `logp` of `z_scale__` with that of a scaled Gaussian:

# %%
print("z_scale__")
model.z_scale__.logp({model.z_scale__:np.ones(4)})

# %%
print("scaled Gaussian")
l = pm.Normal.dist(mu=np.zeros(4), sigma=np.array([1., 2., 3., 4.])).logp(np.ones(4)).eval()
print(l)
print(l.sum())

# %%
for draws in z_draws.T:
    plt.hist(draws, alpha=0.5, bins='auto')


# %% [markdown]
# ### Composing transforms
#
# Example: We want to define a RV $W$ such that its columns sum to 1. Since the Dirichlet distribution can be used to define a RV where rows sum to one, all we need is to take its transpose.

# %%
class TransposeTransform(pm.transforms.Transform):
    name = "transpose"
    
    def backward(self, x):
        return x.T
    def forward(self, x):
        return x.T
    def forward_val(self, x, point=None):
        return x.T
    def jacobian_det(self, x):
        return tt.constant(0, dtype=x.dtype)
transpose = TransposeTransform()

# %%
with pm.Model() as model1:
    W = pm.Dirichlet('W', np.ones((3,3)))

# %%
Wdraw = W.random()
Wdraw_s = np.array([f"{w:.3f}" for w in Wdraw.flat]).reshape(Wdraw.shape)
np.hstack([Wdraw_s, np.array([["|"]]*3), Wdraw.sum(axis=1,keepdims=True)])

# %% [markdown]
# The model below replaces the transform with transposition. Note however that it is still the rows of $W$ which sum to 1, which makes this a bit pointless.
#
# Moreover, we lose the default stick breaking transform this way, which greatly impacts numerics.

# %%
with pm.Model() as model2:
    W = pm.Dirichlet('W__', np.ones((3,3)), transform=transpose)

# %%
Wdraw = W.random()
Wdraw_s = np.array([f"{w:.3f}" for w in Wdraw.flat]).reshape(Wdraw.shape)
np.hstack([Wdraw_s, np.array([["|"]]*3), Wdraw.sum(axis=1,keepdims=True)])

# %%
model1

# %%
model2

# %% [markdown]
# We can check that the transformation performs as expected by applying it directly.
# The problem is just that it is applied “upstream”: the final variable is $W$, which the Dirichlet restricts to having rows which sum to 1.

# %%
# Transformation is transposition
Wdraw = model2.W.distribution.transform.forward_val(Wdraw)
Wdraw_s = np.array([f"{w:.3f}" for w in Wdraw.flat]).reshape(Wdraw.shape)
np.hstack([Wdraw_s, np.array([["|"]]*3), Wdraw.sum(axis=1,keepdims=True)])

# %% [markdown]
# What we want is for $W$ to be some intermediate variable, which would then be transposed in the same way as `W_stickbreaking__`. We try to do this below by replacing `W` with `base_dist`, and constructing a new random variable `W` using lower-level PyMC3 functions, but this is not yet functional.

# %%
with pm.Model() as model3:
    #W_dist = transpose.apply(pm.Dirichlet.dist(np.ones((3,3))))
    #W_dist = transpose.apply(pm.Normal.dist(np.ones((2,3)), shape=(2,3)))
    base_dist = pm.Normal.dist(np.ones((2,3)), shape=(2,3))
    W_dist = transpose.apply(base_dist)
    W = model3.Var('W', W_dist)

# %%
pm.distributions.draw_values([W])

# %%
shim.graph.symbolic_inputs(W)

# %%
pm.sample_prior_predictive(model=model3, samples=1)

# %% [markdown]
# ## Preventing PyMC3 from applying transforms automatically
# This is easily done by passing `transform=None` as keyword argument to a distribution which would otherwise be transformed.

# %%
with pm.Model() as model:
    pm.HalfNormal('a', sigma=1)
    pm.HalfNormal('b', sigma=1, transform=None)

# %%
print("a: ", type(model.a))
print("b: ", type(model.b))
print(model.a.random())
print(model.b.random())

# %%
model
