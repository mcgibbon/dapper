# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:02:22 2016

@author: mcgibbon
"""
from .. import _export as package_export
__all__ = []


def export(defn):
    globals()[defn.__name__] = defn
    __all__.append(defn.__name__)
    defn = package_export(defn)
    return defn

from . import dataset, magic, retrieval, time, util
