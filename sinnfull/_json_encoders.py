# -*- coding: utf-8 -*-

"""
Import sinnfull types and associate them with their JSON encoders.
This module is not meant to be imported directly, but by sinnfull.__init__
(it is separate to prevent import cycles).
"""

from mackelab_toolbox.cgshim import typing as mtbtyping
import mackelab_toolbox.serialize as mtbserialize
import smttask.typing

from .data import DataAccessor
from .parameters import ParameterSet
from .typing_ import IndexableNamespace
# from .models.base import ObjectiveFunction

# Serialization of functions
mtbserialize.config.trust_all_inputs = True
for T, encoder in mtbserialize.json_encoders.items():
    mtbtyping.add_json_encoder(T, encoder)
# Serialization of SmtTask types (some redundancy with mtb.serialize)
for T, encoder in smttask.typing.json_encoders.items():
    mtbtyping.add_json_encoder(T, encoder)
# sinnfull-specific json serializers
mtbtyping.add_json_encoder(ParameterSet, ParameterSet.json_encoder)
mtbtyping.add_json_encoder(IndexableNamespace, IndexableNamespace.json_encoder)
mtbtyping.add_json_encoder(DataAccessor, lambda data: data.to_desc())
# Serialization of quantities with units
mtbtyping.load_pint()
# # Serialization of ObjectiveFunction
# mtbserialize.config.default_namespace.update(ObjectiveFunction=ObjectiveFunction)
# # mtbtyping.add_json_encoder(ObjectiveFunction, ObjectiveFunction.json_encoder)

json_encoders = mtbtyping.json_encoders
