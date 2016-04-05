# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:02:22 2016

@author: mcgibbon
"""
from .. import export as package_export
__all__ = []


def export(defn, as_name=None):
    if as_name is None:
        as_name = defn.__name__
    globals()[as_name] = defn
    __all__.append(as_name)
    defn = package_export(defn, as_name)
    return defn

from . import dataset, magic, retrieval, time, util
