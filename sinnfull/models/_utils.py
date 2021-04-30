"""
Helper functions for displaying, documenting and testing priors.
"""

from __future__ import annotations

import numpy as np
import pymc3 as pm
import theano_shim as shim

import holoviews as hv
hv.extension('bokeh')

# TODO: Display no. of discarded samples with histogram
def truncated_histogram(data, N=1000, α=0.01, **kwargs):
    """
    :param:N: Number of samples
    :param:α: Maximum fraction of discarded samples
    """
    freqs, edges = np.histogram(data, **kwargs)

    α = 0.10  # Reject up to 1% of samples
    N = 1000
    i = 1
    ubound = edges[-i]
    t = freqs[-1]
    while t < (α/2)*N:
        i += 1
        ubound = edges[-i]
        t += freqs[-i]
    i = 0
    lbound = edges[i]
    t = freqs[0]
    while t < (α/2)*N:
        i += 1
        lbound = edges[i]
        t += freqs[i]
    freqs, edges = np.histogram(data, range=(lbound,ubound), **kwargs)
    return freqs, edges

def sample_prior(prior, N=1000) -> hv.Layout:
    """
    A simple function which displays a grid of marginals from the given prior.
    Useful to visualize the parameter ranges it specifies.
    """
    draws = pm.sample_prior_predictive(model=prior, samples=N)
    hists = []
    for θname, θdraws in draws.items():
        if not shim.graph.symbolic_inputs(getattr(prior, θname)):
            # Deterministic constants end up here
            # TODO: Should we return something ? A table of contants ?
            # NB: Can't just do ndim==0, because θ could be a vector
            continue
        elif θdraws.ndim == 1:
            edges, frequencies = truncated_histogram(θdraws, bins='auto')
            hists.append(hv.Histogram((edges, frequencies), kdims=[θname]))
        else:
            θdraws = θdraws.reshape(len(θdraws),-1)
            for i in range(θdraws.shape[1]):
                edges, frequencies = truncated_histogram(θdraws[:,i], bins='auto')
                hists.append(hv.Histogram((edges, frequencies), kdims=[f"{θname}_{i}"]))

    return hv.Layout(hists).opts(hv.opts.Histogram(width=200, height=100, axiswise=True)).cols(4)
