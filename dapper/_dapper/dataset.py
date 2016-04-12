# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:17:33 2016

@author: mcgibbon
"""
import xarray
from six import string_types
from datetime import timedelta
from .time import TimeRange, get_netcdf_time, datetime
from . import export
from .util import zlcl_from_T_RH
from .magic import leg_times
import numpy as np
import re
import atmos
try:
    from numba import jit
except ImportError:
    def jit(function):
        return function


def align(*objects, **kwargs):
    """Given any number of Dataset objects, returns new
    objects with aligned indexes.

    Array from the aligned objects are suitable as input to mathematical
    operators, because along each dimension they have the same indexes.

    Missing values (if ``join != 'inner'``) are filled with NaN.

    Parameters
    ----------
    *objects : Dataset
        Objects to align.
    join : {'outer', 'inner', 'left', 'right'}, optional
        Method for joining the indexes of the passed objects along each
        dimension:
        - 'outer': use the union of object indexes
        - 'inner': use the intersection of object indexes
        - 'left': use indexes from the first object with each dimension
        - 'right': use indexes from the last object with each dimension
    copy : bool, optional
        If ``copy=True``, the returned objects contain all new variables. If
        ``copy=False`` and no reindexing is required then the aligned objects
        will include original variables.

    Returns
    -------
    aligned : same as *objects
        Tuple of objects with aligned coordinates.
    """
    xarray_datasets = [obj.xarray for obj in objects]
    aligned_datasets = xarray.align(xarray_datasets, **kwargs)
    return [Dataset(ds) for ds in aligned_datasets]


class EmptyDataset(object):
    pass


@export
class Dataset(object):

    def __init__(self, xarray_dataset, variable_aliases=None):
        """
        variable_aliases should be a dictionary whose keys are aliases, and values
        are the variable names they refer to.
        """
#        if 'time_offset' in xarray_dataset:
#            xarray_dataset['time'] = xarray_dataset['time_offset']
        self._dataset = xarray_dataset
        self._time = get_netcdf_time(self._dataset)
        if not hasattr(self, 'variable_aliases'):  # subclass might initialize this
            self.variable_aliases = {}
        if variable_aliases is not None:
            for alias, variable_name in variable_aliases.items():
                self.define_alias(alias, variable_name)

    def __repr__(self):
        return self._dataset.__repr__()

    def __str__(self):
        return self._dataset.__str__()

    def __unicode__(self):
        return self._dataset.__unicode__()

    def resample(self, window='3H', label='center'):
        if label == 'center':
            xarray_dataset = self._dataset.resample(window, 'time', how='mean', label='left')
            time_offset_to_middle = 0.5*(xarray_dataset['time'][1] - xarray_dataset['time'][0])
            xarray_dataset['time'] += time_offset_to_middle
        else:
            xarray_dataset = self._dataset.resample(window, 'time', how='mean', label=label)
        return Dataset(xarray_dataset)

    def __getitem__(self, key):
        if key in self.variable_aliases:
            return self[self.variable_aliases[key]]
        elif key in self._dataset:
            return self._dataset[key]
        else:
            raise KeyError()

    def __setitem__(self, key, item):
        self._dataset[key] = item

    def define_alias(self, alias, variable_name):
        if not self._dataset_has_variable(variable_name):
            raise ValueError(
                'Cannot create alias for non-existent variable {}'.format(variable_name))
        else:
            self.variable_aliases[alias] = variable_name

    def _dataset_has_variable(self, variable_name):
        return variable_name in self._dataset.data_vars.keys()

    @property
    def time(self):
        return self._time.values

    @property
    def time_range(self):
        return TimeRange(self.time[0], self.time[-1])

    @property
    def xarray(self):
        return self._dataset

