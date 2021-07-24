import logging
from functools import total_ordering
from typing import Union, Any, Tuple
import holoviews as hv

from .config import pretty_names

logger = logging.getLogger(__name__)

__all__ = ['StrTuple']

@total_ordering
class StrTuple(str):
    """
    Tuple substitute used to create tuple keys usable in cases where a string
    is needed. Motivating use case: tuples don't work well as HoloMap keys,
    because they are used by `select` to select a range of values.

    All of these instantiations are equivalent:

    >>> StrTuple(5, 33)
    >>> StrTuple((5, 33))
    >>> StrTuple("(5,33)")
    >>> StrTuple("(5, 33)")

    Differences wrt to strings:

    - Order like tuples: '(5, 10)' > '(5, 3)'
    - Len like tuples: len('(5, 10)') == 2
    - Indexable like tuples (always return StrTuple): '(5, 10)'[1] == '(10)'
    """
    def __new__(cls, *args):
        # Special case for instantiating from a tuple string
        if len(args) == 1 and cls._looks_like_tuple(args[0]):
            if isinstance(args[0], tuple):
                return super().__new__(cls, args[0])
            else:
                assert isinstance(args[0], str)
                els = list(filter(None, cls._split_tuple_str(args[0])))
                tup_s = ', '.join(els)
                if len(els) == 1:
                    tup_s += ','
                return super().__new__(cls, f'({tup_s})')
        elif len(args) > 1:
            # Wrap multiple arguments into a single tuple and use the branch above
            return StrTuple.__new__(cls, args)
        else:
            # Normal case
            return super().__new__(cls, *args)
    # Adaptation of str interface
    def split(self):
        raise NotImplementedError
    def __hash__(self):
        return hash(str(self))
    # Tuple-like interface
    def __len__(self) -> int:
        return sum(bool(x) for x in self._split())
    # Comparisons
    @staticmethod
    def _looks_like_tuple(s: Any) -> bool:
        if isinstance(s, tuple):
            return True
        elif not isinstance(s, str):
            return False
        s = s.strip()
        if len(s) == 2:
            return s == "()"
        else:
            return (s[0] == '(' and s[-1] == ')' and ',' in s)
    @staticmethod
    def _split_tuple_str(s: str):
        return (x.strip() for x in s.strip('()').split(','))
    def _split(self):
        return self._split_tuple_str(self)
    def __eq__(self, other: Any):
        if isinstance(other, str):
            return all(x_self == x_other
                       for x_self, x_other
                       in zip(self._split(), self._split_tuple_str(other)))
        elif isinstance(other, tuple):
            return (len(other) == len(self)
                    and all(x_self == str(x_other).strip()
                            for x_self, x_other in zip(self._split(), other)))
        elif isinstance(other, StrTuple):
            return self == other
        else:
            return NotImplemented
    def __lt__(self, other: Any):
        if isinstance(other, str) and self._looks_like_tuple(other):
            other_els = self._split_tuple_str(other)
        elif isinstance(other, tuple):
            other_els = (str(x) for x in other)
        elif isinstance(other, StrTuple):
            other_els = other._split()
        else:
            return NotImplemented
        for x_self, x_other in zip(self._split(), other_els):
            # Prepend with zeros so numbers compare correctly
            L = max(len(x_self),len(x_other))
            x_self = f"{x_self:0>{L}}"
            x_other = f"{x_other:0>{L}}"
            # Compare lexicographically. This works on numbers because we padded
            if x_self < x_other:
                return True
            elif x_self > x_other:
                return False
        return False  # We reach this point if self == other

    # The following perfectly emulates tuple slicing, BUT breaks code which
    # expects subtypes of str to slice like strings (incl. a hv.core.utils.closest_match)
    # def __getitem__(self, key: Union[int,slice]):
    #     tup = tuple(self._split())[key]
    #     tup_s = ', '.join(x for x in tup)
    #     if len(tup) == 1:
    #         tup_s += ','
    #     return StrTuple(f'({tup_s})')

class KeyDimensions(dict):
    """
    A dynamically created dictionary of `holoviews.Dimensions` instances for
    collection keys; dimensions are keyed by their name.
    Dimensions are created the first time they are accessed: thus the
    dictionary does not have to be populated in advance, and subsequent
    accesses are guaranteed to retrieve the same dimension instance.

    The default `pretty_names` dictionary is used to set the dimension label.
    """
    def get(self, dim_name: str, *, label=None) -> hv.Dimension:
        if isinstance(dim_name, hv.Dimension):
            if label is not None and dim_name.name != dim_name.label:
                raise TypeError("Specifying both a full Holoviews Dimension "
                                "and a label is ambiguous.")
            label = dim_name.label
            dim_name = dim_name.name
        try:
            dim = self[dim_name]
        except KeyError:
            if label is None:
                label = pretty_names.wrap(pretty_names.get(dim_name))
            dim = hv.Dimension(dim_name, label=label)
            self[dim_name] = dim
        else:
            if label is not None and dim.name == dim.label and dim_name != label:
                # Heuristic: Assume that name == label implies that no label was assigned
                logger.debug(f"Assigning label '{label} to dimension "
                             f"'{dim.name}' because it had no previous label.")
                self[dim_name].label = label
            elif label is not None and dim.label != label:
                # Create a throwaway Dimension with the different label
                logger.warning(f"Memoized Dimension '{dim_name}' has a different "
                               f"label ('{dim.label}' instead of '{label}'). "
                               "Creating a throwaway Dimension object...")
                dim = hv.Dimension(dim_name, label=label)
        return dim

class ParamDimensions(dict):
    """
    A dynamically created dictionary of `holoviews.Dimensions` instances for
    parameters; dimensions are keyed by parameter name and index.
    Dimensions are created the first time they are accessed: thus the
    dictionary does not have to be populated in advance, and subsequent
    accesses are guaranteed to retrieve the same dimension instance.

    The default `pretty_names` dictionary is used to set the dimension label.
    """
    _num_subscripts = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    def __init__(self, key_dims: KeyDimensions):
        self.key_dims = key_dims

    def get(self, θname: str, θidx: Union[tuple,str]) -> hv.Dimension:
        idx_els = [i.strip() for i in str(θidx).strip('(),').split(',')]  # Same result with both tuples and strings
        if any(len(i) > 1 for i in idx_els):
            raise NotImplementedError(
                "ParamDimensions would be ambiguous with dimension indices "
                "> 10. The code could be updated to add a comma in these cases.")
        if len(idx_els):
            idx_short_str = ''.join(idx_els)
            idx_long_str = pretty_names.index_str(idx_els)
        else:
            idx_short_str = idx_long_str = ""
        try:
            dim = self[(θname, idx_short_str)]
        except KeyError:
            dim = self.key_dims.get(
                θname+idx_short_str,
                label=pretty_names.wrap(pretty_names.get(θname)+idx_long_str))
            self[(θname, idx_short_str)] = dim
        return dim
