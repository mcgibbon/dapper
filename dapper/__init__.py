# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 11:02:22 2016

@author: mcgibbon
"""
__all__ = []


def export(defn, as_name=None):
    if as_name is None:
        as_name = defn.__name__
    globals()[as_name] = defn
    __all__.append(as_name)
    return defn

from ._dapper import *
