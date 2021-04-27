"""
A small set of patches on the *parameters* packages.

Import ParameterSet through here, to make sure patches are applied.

Effect:
  - Replace the Parameter class by a function returning a plain value, but uses
    the `type` and `units` cast and apply a (Pint) unit as appropriate.
"""

# In order to allow this to be run from notebooks within the same directory,
# we need to allow a bit of Python path hacking:
# There are two import branches:
# `import parameters` should return the base `parameters`
# `import sinnfull` should return this package
# If `import parameters` is run from the same directory as this file, we end
# end up here instead of the `parameters` package. This is detected by checking
# `__name__`, and then the 'parameters' entry is removed from sys.modules
# to make space for the real one.
import sys
from warnings import warn
from mackelab_toolbox.meta import HideCWDFromImport

if __name__ == 'parameters':
    # Imported as `import parameters`, not `import sinnfull.parameters`
    # Rename in sys.modules to make space for the base module `parameters`
    assert 'parameters' in sys.modules
    del sys.modules['parameters']
    with HideCWDFromImport(__file__):
        import parameters
    # We don't need to do anything more: importlib will return whatever is in
    # sys['parameters'] at the end of its return.
else:
    from copy import deepcopy
    from typing import Tuple
    from functools import reduce
    import numpy as np
    import pint
    import mackelab_toolbox.parameters
    import theano_shim as shim
    from sinn.models import ModelParams
    from sinnfull import ureg
    with HideCWDFromImport(__file__):  # Ensure the parameters package is not shadowed by this module
        import parameters as _parameters
        from parameters import *  # Make this a drop-in replacement for parameters

    # Monkey patch Parameter so that it applies type and units rather than just
    # storing them.
    BaseParameter = _parameters.Parameter
    def Parameter(value, units=None, name="", type=None):
        if type is not None:
            value = np.dtype(type).type(value)
        p = BaseParameter(value, units=units, name=name)
        val = p.value
        if p.units is not None:
            val = p.value * ureg(p.units)
        return val
    ParameterSet.namespace['Parameter'] = Parameter

    # def load_learning_parameters(url):
    #     learn_params = ParameterSet(url)
    #     # Wrap learning parameters with shared variables
    #     for nm, val in learn_params.flat():
    #         if nm in ('Adamθ.lr', 'λη'):
    #             learn_params[nm] = shim.shared(val)
    #     return learn_params

    def apply_sgd_masks(*paramsets: Tuple[ModelParams, dict],
                        on_non_shared_param: str='fit',
                        str_keys: bool=True):
        """
        For each passed pair of (parameters,mask), extract the parameter
        components for which the corresponding mask is True.

        Parameters
        ----------
        *paramsets (parameters, masks) pairs.
            `parameters` should be instance of a Model.Parameters.
            `mask` is a dictionary of 'param name': bool (or Array[bool]).
            The param names in `mask` must occur within `parameters`.
        on_non_shared_param: 'fit' | 'raise' | 'warn' | 'ignore'
            What to do if some parameters are not shared variables ?
            (Those model parameters which are not shared vars cannot be optimized.)
            Default is to save a mask regardless; this is appropriate when
            the paramset is the initialization vector, rather than the model
            parameters themselves.
            Requires `str_keys` to be ``True``.
            'raise', 'warn' and 'ignore' will all skip application of the mask
            for non-shared parameters; this is appropriate when the model is
            already in memory, and `paramset` is a `ModelParams` instance.
        str_keys: bool
            ``True``: Before returning the dictionary, shared variables are
            replaced by their attribute name in the param_set

        :returns: dict {param: mask}
            `param` is either a string or shared variable,
            `mask` a bool or Array[bool]
        """
        Θ = {}  # Flattened vector of parameters to optimize
        non_shared_θ = {}  # Should remain empty; raises TypeError otherwise
        if not str_keys and on_non_shared_param == 'fit':
            raise ValueError("When `on_non_shared_param` = 'fit', then "
                             "`str_keys` must be True.")
        for θ_set in paramsets:
            if isinstance(θ_set, tuple):
                θ_set, masks = θ_set
            else:
                # Default to fitting everything
                masks = {θ: True for θ in θ_set}
            for nm, mask in masks.items():
                θ = getattr(θ_set, nm)
                if mask is not False:
                    if not shim.isshared(θ) and on_non_shared_param != 'fit':
                        non_shared_θ[nm] = θ
                    elif str_keys:
                        Θ[nm] = mask
                    else:
                        assert shim.isshared(θ)
                        Θ[θ] = mask
        if non_shared_θ:
            msg = ("You asked to optimize the following values, but they are "
                   f"not shared variables:\n{non_shared_θ}")
            if on_non_shared_param == 'ignore':
                pass
            elif on_non_shared_param == 'warn':
                warn(msg)
            else:
                raise TypeError(msg)
        if len(Θ) == 0:
            warn("`apply_sgd_masks` returned an empty dict. No parameter will "
                 "be optimized.")
        return Θ

    #TODO: Generalize accept quantities units as well, and move to mackelab_toolbox.units.
    def to_ref_units(value, ref_units: dict):
        """
        Similar to pint's `to_base_units`, where the base units are specified
        by `ref_units`.

        :params value: Unit value to convert.
        :params ref_units: Dict of '[dimension]': unit pairs.
            `unit` can be anything accepted by `pint.UnitRegistry`.

        """
        dims = getattr(value, 'dimensionality', None)
        # Exit if e.g. `value` is a plain float
        if dims is None:
            return value
        # Normalize the `ref_units` dictionary
        ref_units = ref_units.copy()
        for nm, unit in ref_units.items():
            if not isinstance(unit, ureg.Quantity):
                ref_units[nm] = ureg(unit)
        # Figure out the target unit from the dimensionality
        target_units = reduce(lambda u,d: u*ref_units[d[0]]**d[1], value.dimensionality.items(), 1)
        # target_units = np.prod([ref_units[d]**p for d,p in value.dimensionality.items()])
        return value.to(target_units)

    class ParameterSet(_parameters.ParameterSet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Re-iterate through the dictionary `d`, replacing base `ParameterSet`
            # with our customized `ParameterSet` object.
            # This ensures that all nested ParameterSet also use the customized type.
            def walk(d):
                for k, v in d.items():
                    if (not isinstance(v, ParameterSet)
                        and isinstance(v, dict)):
                        d[k] = walk(v)
                if not isinstance(d, ParameterSet):
                    d = ParameterSet(d)
                return d
            for k, v in self.items():
                if  (not isinstance(v, ParameterSet)
                     and isinstance(v, dict)):
                    self[k] = walk(v)
            # If there is a 'unit' entry, replace the string names by actual units
            # This is only applied to keys in the '[dimension]' format.
            if 'units' in self:
                units = self.units
                for dim, unit in units.items():
                    if dim.startswith('[') and dim.endswith(']'):
                        if isinstance(unit, (ureg.Quantity, ureg.Unit)):
                            pass
                        elif isinstance(unit, (list, tuple)):
                            if len(unit) == 2 and unit[0] == "PintValue":
                                # A bit of a hack: to avoid adding a dependency
                                # on mackelab_toolbox.typing, we hard code the
                                # serialization format it uses
                                unit = unit[1]
                            unit = ureg.Quantity.from_tuple(unit)
                        else:
                            unit = ureg(unit)
                            units[dim] = unit
                        assert len(unit.dimensionality) == 1  # Unclear yet if we should deal with compound units
                        assert next(iter(unit.dimensionality)) == dim

        def remove_units(self, ref_units):
            """
            Remove units on values, converting them first to a set of
            reference units.

            If any value is a string, attempts to convert to a pint.Quantity
            before converting to the reference units.
            """
            for nm, value in self.flat():
                if isinstance(value, str):
                    try:
                        value = ureg.Quantity(value)
                    except pint.UndefinedUnitError:
                        pass
                if nm.startswith('units'):
                    # Don't strip the units from the ref units :-p
                    pass
                elif hasattr(value, 'dimensionality'):
                    self[nm] = to_ref_units(value, ref_units).magnitude

        def outer_product(self, params=None):
            """
            This method can be used to convert a ParameterSet with
            ParameterRange elements into a list of parameter sets.
            'Outer_product' corresponds to nested loops over each Range element.
            Like `inner_product`, this yields an iterator, and either method
            can be chained with nested loops.
            """
            params_to_iter = []
            for k, v in self.flat:
                if isinstance(v, ParameterRange):
                    if params is None or k in params:
                        params_to_iter.append(k)
            val_iter = itertools.product(*(self[k] for k in params_to_iter))
            for vals in val_iter:
                ps = deepcopy(self)
                for k, v in zip(params_to_iter, vals):
                    ps[k] = v
                yield ps

        def inner_product(self, params=None):
            """
            This method can be used to convert a ParameterSet with
            ParameterRange elements into a list of parameter sets.
            'Inner_product' corresponds to zipping simultaneouslys over each Range element.
            Like `outer_product`, this yields an iterator, and either method
            can be chained with nested loops.
            """
            params_to_iter = []
            for k, v in self.flat:
                if isinstance(v, ParameterRange):
                    if params is None or k in params:
                        params_to_iter.append(k)
            if len(params_to_iter) == 0:
                warn("ParameterSet.inner_product: No parameters to iterate")
                yield self
                return
            else:
                n_vals = len(self[params_to_iter[0]])
                for k in params_to_iter:
                    if len(self[k]) != n_vals:
                        warn("ParameterSet.inner: Not all parameter ranges "
                             "have the same length.")
            val_iter = zip(*(self[k] for k in params_to_iter))
            for vals in val_iter:
                ps = deepcopy(self)
                for k, v in zip(params_to_iter, vals):
                    ps[k] = v
                yield ps

        def diff(self, other):
            # Use the patched _dict_diff, which deals correctly with numpy
            # arrays, and different dict hierarchies (Sumatra's _dict_diff
            # fails on {'a': {'b': 1}} vs {'a': 1}
            return mackelab_toolbox.parameters._dict_diff(self, other)
