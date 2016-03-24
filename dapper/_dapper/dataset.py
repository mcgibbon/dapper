# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:17:33 2016

@author: mcgibbon
"""
import xarray
from six import string_types
from .time import TimeRange


def get_netcdf_time(dataset, time_varname='time_offset'):
    return dataset.variables[time_varname].values


class EmptyDataset(object):
    pass


class Dataset(object):

    def __init__(self, xarray_dataset, variable_aliases=None):
        """
        variable_aliases should be a dictionary whose keys are aliases, and values
        are the variable names they refer to.
        """
        self._dataset = xarray_dataset
        self.time = get_netcdf_time(self._dataset)
        self.variable_aliases = {}
        if variable_aliases is not None:
            for alias, variable_name in variable_aliases.items():
                self.define_alias(alias, variable_name)

    def resample(self, window='3H'):
        return Dataset(self._dataset.resample(window, 'time', how='mean'))

    def __getitem__(self, key):
        if key in self.variable_aliases:
            return self[self.variable_aliases[key]]
        else:
            return self._dataset[key]

    def define_alias(self, alias, variable_name):
        if not self._dataset_has_variable(variable_name):
            raise ValueError(
                'Cannot create alias for non-existent variable {}'.format(variable_name))
        else:
            self.variable_aliases[alias] = variable_name

    def _dataset_has_variable(self, variable_name):
        return variable_name in self._dataset.data_vars.keys()

    @property
    def time_range(self):
        return TimeRange(self.time[0], self.time[-1])

    @property
    def xarray(self):
        return self._dataset


class FilenameDataset(Dataset):

    def __init__(self, filenames):
        if isinstance(filenames, string_types):
            self._dataset = xarray.open_dataset(filenames)
        else:
            self._dataset = xarray.open_mfdataset(filenames)
        self.time = get_netcdf_time(self._dataset)
        self.variable_aliases = {}
