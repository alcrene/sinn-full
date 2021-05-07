# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#   kernelspec:
#     display_name: Python (comp)
#     language: python
#     name: comp
# ---

# # Tagged collections
#
# See also [models – Taming proliferation](tags-taming-model-proliferation).

from __future__ import annotations
from collections.abc import Iterable, Sized
from typing import Union, Dict

class TaggedCollection(list):
    tags: set
    values: list
    _squeeze: bool  # Whether to return a plain list when the set of tags is empty,
                    # and an single element when the collection as length 1.
                    # Applies only to __getitem__ and __getattr__
                    # Default: True

    @classmethod
    def from_tagged_objects(cls, iterable, tag_attr='_tags', if_tag_attr_missing='skip'):
        """
        Create a `TaggedCollection` from objects which have been tagged by a
        `TagDecorator` – i.e. have an attribute which stores a set of strings.
        Default attribute name is '_tags', same as `TagDecorator.
        By default, if an item in `iterable` does not contain an attribute
        matching `tag_attr`, it is silently skipped. To raise an `AttributeError`
        instead, pass ``if_tag_attr_missing='raise'``.
        """
        if if_tag_attr_missing == 'skip':
            gen = ((x, getattr(x,tag_attr,None)) for x in iterable)
            return cls(x_tags for x_tags in gen if x_tags[1] is not None)
        else:
            return cls((x, getattr(x,tag_attr)) for x in iterable)
    @staticmethod
    def _set_cast_str(tags):
        if isinstance(tags, str):
            return {tags}
        else:
            return set(tags)
    def __init__(self, iterable: Iterable, squeeze=True):
        super().__init__((x, self._set_cast_str(tags)) for x, tags in iterable)
        # Set self.tags, self.values, and check for duplicates
        self.validate_tags()
        # Set self._squeeze
        self._squeeze = squeeze

    def __str__(self):
        return f"TaggedCollection<{len(self)} items, tags={self.tags}>"
    def by_tag(self) -> Dict[frozenset, list]:
        tag_sets = {}
        for obj, tags in self:
            tag_sets[frozenset(tags)] = obj
        return tag_sets

    def append(self, value):
        "Note: This function revalidates the entire list of tags on each call."
        assert isinstance(value, Sized) and len(value) == 2
        super().append(value)
        self.validate_tags()

    def validate_tags(self):
        ## Set self.tags
        self.tags = set().union(*(tags for x,tags in self))
        if not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("All tags must be strings. Offending tags: "
                             f"{[tag for tag in self.tags if not isinstance(tag, str)]}")
        ## Set self.values
        # Find duplicates
        # by_id = {id(x):(i,x) for i,(x,tag) in enumerate(self)}
        seen = {}
        dups = {}
        for x,tags in self:
            if id(x) in seen:
                if id(x) not in dups:
                    dups[id(x)] = [seen[id(x)][1], tags]
                else:
                    dups[id(x)].append(tags)
            else:
                seen[id(x)] = (x,tags)
        invalid_dups = [(seen[xid][0],tag_sets) for xid,tag_sets in dups.items()
                        if any(tag_set != tag_sets[0] for tag_set in tag_sets)]
        if invalid_dups:
            raise ValueError("Multiple sets of tags were specified for the "
                             f"following values: {invalid_dups}.")
        # Remove duplicates in `self` by replacing with values in `seen.values()`
        self[:] = seen.values()

    @property
    def values(self):
        return [x for x,tags in self]

    def filter(self, tag, remove=False):
        """
        Return a TaggedCollection containing only the values matching the
        provided tag(s).
        If `tag` is a `set`, it should contain multiple tags. They must all
        match for an element to be kept by the filter.
        If `remove=True`, matching tags are removed from the elements of the
        returned collection.
        """
        if isinstance(tag, set):
            filter_tags = tag
        else:
            filter_tags = {tag}
        remove_set = filter_tags if remove else {}
        return TaggedCollection((x, tags-remove_set) for x, tags in self
                                if filter_tags <= tags)
    def filter_not(self, tag):
        """
        Return a TaggedCollection containing only the values NOT matching the
        provided tag(s).
        If `tag` is a `set`, it should contain multiple tags. None of them must
        match for an element to be kept by the filter.
        """
        if isinstance(tag, set):
            filter_tags = tag
        else:
            filter_tags = {tag}
        return TaggedCollection((x, tags) for x, tags in self
                                if filter_tags.isdisjoint(tags))

    def _apply_squeeze(self, result):
        if self._squeeze and len(result) == 1:
            res = result[0]
            # HACK - Attach the remaining tags to the result, so that
            # overspecifying tags does not cause an error.
            # Attributes simply point back to the object, so that indexing
            # redundant tags is a noop
            # if result.tags:
            #     for tag in result.tags:
            #         if hasattr(res, '_selfrefs'):
            #             # `res` provides a dictionary for self-references;
            #             # use that to store the additional tags.
            #             # (self-references are used as fallbacks for __getattr__)
            #             res._selfrefs.add(tag)
            #         # if not hasattr(res, tag):  # Don't overwrite existing attributes
            #         #     try:
            #         #         setattr(res, tag, res)
            #         #     except Exception:
            #         #         # This is just a hacky convenience.
            #         #         # If it fails for any reason, continue.
            #         #         continue
            return res
        elif self._squeeze and len(result.tags) == 0:
            return result.values
        else:
            return result

    def __getattr__(self, attr):
        if attr in self.tags:
            return self._apply_squeeze(self.filter(attr, remove=True))
        else:
            raise AttributeError(f"'{attr}' does not match any attribute or tags. "
                                 f"Possible tags: {self.tags}.")
    def __getitem__(self, key):
        """
        Two main interpretations of `key`
          * Filter (key is str or Set[str])
            -> return a filtered TaggedCollection
          * Index (key is int or slice)
            -> return an element (int) or list (slice)
        """
        # Allow passing multiple tags as comma separated list
        if isinstance(key, tuple):
            key = set(key)
        if isinstance(key, str):
            if key in self.tags:
                return self._apply_squeeze(self.filter(key, remove=True))
            elif key[1:] in self.tags:
                mod, key = key[0], key[1:]
                if mod == "!":
                    return self._apply_squeeze(self.filter_not(key))
                else:
                    raise ValueError(f"'{mod}' is not a recognized tag "
                                     "modifier. Possible values: !.")
            else:
                raise KeyError(f"'{key}' does not match any tags. "
                               f"Possible tags: {self.tags}.")
        elif isinstance(key, set):
            # TODO: Support multiple negation filters
            if key - self.tags:
                raise KeyError("The following values don't match any tags: "
                               f"{key-self.tags}.\nPossible tags: {self.tags}.")
            return self._apply_squeeze(self.filter(key, remove=True))
        elif isinstance(key, int):
            return super().__getitem__(key)[0]
        elif isinstance(key, slice):
            return [x[0] for x in super().__getitem__(key)]
        else:
            raise TypeError(f"Invalid key type {type(key)} – should be str, "
                            f"Set[str], int or slice\nReceived: {key}.")
    def __contains__(self, key):
        return key in self.values

class TagDecorator:
    def __init__(self, attribute_name: Union[str,Dict[type,str]]='_tags'):
        """
        :param:attribute_name: The attribute name used to attach tags.
            Can also be a dictionary, mapping types to strings.
            The `sinnfull.utils.TypeDict` is meant for this purpose.
        """
        self.attr_name = attribute_name
    def __call__(self, *tags):
        def decorator(obj):
            attr_name = self.attr_name
            if isinstance(attr_name, dict):
                attr_name = attr_name[obj]
            tag_attr = getattr(obj, attr_name, None)
            if tag_attr is None:
                setattr(obj, attr_name, set())
                tag_attr = getattr(obj, attr_name)
            elif not isinstance(tag_attr, set):
                raise RuntimeError(f"Object {obj} already has an attribute "
                                   f"named '{attr_name}'. Cannot use that "
                                   "name to store tags.")
            tag_attr.update(tags)
            return obj
        return decorator
    def __getattr__(self, attr):
        if attr == self.attr_name:
            raise AttributeError(f"{attr} cannot be used as an tag name.")
        return self(attr)

if __name__ == "__main__":
    coll = TaggedCollection([(set('abcdef'), {'alpha', 'contiguous'}),
                             (set([1,2,3,4,5,6]), {'num', 'contiguous'}),
                             ([1,2,30,41,100], 'num')])

    assert coll.tags == {'alpha', 'contiguous', 'num'}
    assert coll[0] == {'a', 'b', 'c', 'd', 'e', 'f'}
    assert coll.num.tags == {'contiguous'}
    assert coll.num.contiguous == {1, 2, 3, 4, 5, 6}  # squeeze, len 1 => single element
    assert len(coll.num) == 2
    assert coll.alpha == {'a', 'b', 'c', 'd', 'e', 'f'}  # squeeze, len 1 => single element
    assert len(coll.contiguous) == 2

    tag = TagDecorator()

    @tag.alpha
    @tag.num
    class Foo:
        pass

    @tag('alpha', 'num')
    class Bar:
        pass

    @tag.alpha
    def Baz(x):
        return x

    coll2 = TaggedCollection.from_tagged_objects((Foo, Bar, Baz))
    assert len(coll2) == 3
    assert coll2.tags == {'alpha', 'num'}
    assert Foo in coll2.num
