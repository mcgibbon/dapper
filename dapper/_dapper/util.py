# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:18:26 2016

@author: mcgibbon
"""
from xarray.ufuncs import log as xarray_log


def zlcl_from_T_RH(T, RH):
    g0 = 9.80665  # standard gravitational acceleration (m/s)
    Cpd = 1005.  # Specific heat of dry air at constant pressure (J/kg/K)
    Gammad = g0/Cpd  # Dry adabatic lapse rate (K/m)
    Tlcl = 1./((1./(T-55.))-(xarray_log(RH/100.)/2840.)) + 55.
    zlcl = (T - Tlcl)/Gammad
    return zlcl
