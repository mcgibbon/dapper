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
try:
    from numba import jit
except ImportError:
    def jit(function):
        return function


class EmptyDataset(object):
    pass


@export
class Dataset(object):

    def __init__(self, xarray_dataset, variable_aliases=None):
        """
        variable_aliases should be a dictionary whose keys are aliases, and values
        are the variable names they refer to.
        """
        if 'time_offset' in xarray_dataset:
            xarray_dataset['time'] = xarray_dataset['time_offset']
        self._dataset = xarray_dataset
        self._time = get_netcdf_time(self._dataset)
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

    def resample(self, window='3H'):
        xarray_dataset = self._dataset.resample(window, 'time', how='mean', label='left')
        time_offset_to_middle = 0.5*(xarray_dataset['time'][1] - xarray_dataset['time'][0])
        xarray_dataset['time'] += time_offset_to_middle
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
        return self._time

    @property
    def time_range(self):
        return TimeRange(self.time[0], self.time[-1])

    @property
    def xarray(self):
        return self._dataset


class FilenameDataset(Dataset):

    def __init__(self, filenames, variable_aliases=None):
        if isinstance(filenames, string_types):
            dataset = xarray.open_dataset(filenames)
        else:
            dataset = xarray.open_mfdataset(filenames)
        super(FilenameDataset, self).__init__(dataset, variable_aliases)


@export
class SamDataset(FilenameDataset):

    def __init__(self, filename, variable_aliases=None):
        if not isinstance(filename, string_types):
            raise NotImplementedError('SamDataset does not work as multi-file dataset')
        super(SamDataset, self).__init__(filename, variable_aliases)
        self._year = get_sam_year_from_filename(filename)
        self._derive_quantities()

    def _derive_quantities(self):
        self['stratocumulus_cbh'] = (['time'], _get_sam_stratocumulus_cbh(self), {'units': 'm'})
        self['LCL'] = (['time'], _get_sam_lcl(self), {'units': 'm'})
        self['z_inv'] = (['time'], _get_sam_z_inv(self), {'units': 'm'})
        delta_q_bl, frac_q_bl = _get_sam_q_bl_products(self)
        self['delta_q_bl'] = (['time'], delta_q_bl, {'units': 'g/kg'})
        self['frac_q_bl'] = (['time'], frac_q_bl, {'units': 'g/kg'})

    @property
    def time(self):
        return day_in_year_to_datetime(self._dataset['time'], self._year)


def get_sam_year_from_filename(filename):
    prog = re.compile(
        r'MAG(?:2|3)D\.(\d+A|\d+B)\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)_\d+h_'
        r'(?:.+_)?\d+x\d+x\d+_LES.nc')
    return leg_times[prog.finditer(filename).next().group(1)][0].year


def day_in_year_to_datetime(day_in_year, year):
    '''Takes in a readable netCDF4 dataset containing output from SAM, and
       an int year corresponding to that data (since SAM only outputs day
       within year).
       Returns a numpy datetime array for the time axis of that dataset,
       assuming UTC timezone.
    '''
    return np.array([(datetime(year, 1, 1) + timedelta(days=float(t)))
                     for t in day_in_year])


def _get_sam_q_bl_products(dataset):
    z_inv = dataset['z_inv'].values
    q = dataset['QT'].values
    z = dataset['z'].values
    q_top = get_values_at_heights(q, height_axis=z, height_values=z_inv - 100.)
    q_near_surface = get_values_at_heights(q, z, np.zeros_like(z_inv) + 100.)
    return q_near_surface - q_top, q_near_surface/q_top


def _get_sam_z_inv(dataset):
    return heffter_pblht(dataset['z'].values, dataset['THETA'].values)


@jit(nopython=True)
def get_values_at_heights(array, height_axis, height_values):
    if len(height_values) != array.shape[0]:
        raise ValueError('must have one height for each point in time')
    output_array = np.zeros((array.shape[0],), dtype=array.dtype)
    for i in range(len(height_values)):
        current_height = height_values[i]
        min_index = 0
        min_distance = abs(height_axis[0] - current_height)
        for j in range(len(height_axis)):
            current_distance = abs(height_axis[j] - current_height)
            if current_distance < min_distance:
                min_index = j
                min_distance = current_distance
        output_array[i] = array[i, min_index]
    return output_array


def heffter_pblht(z, theta):
    """
    Given height and theta (assuming increasing height with index), returns
    the planetary boundary layer height from the Heffter criteria.
    If 1-D, assumes the axis is height and returns a float. If 2-D, assumes
    (time, height) and returns a 1-D array in time.
    """
    if z.shape != theta.shape:
        try:
            z, theta = np.broadcast_arrays(z, theta)
        except ValueError:
            ValueError('must be able to broadcast z and theta')
    if len(z.shape) == 1:  # height axis
        return heffter_pblht_1D(z, theta)
    elif len(z.shape) == 2:
        return_array = np.zeros((z.shape[0],))
        for i in range(z.shape[0]):
            return_array[i] = heffter_pblht_1D(z[i, :], theta[i, :])
        return return_array
    else:
        raise ValueError('data has an invalid number of dimensions')


def heffter_pblht_1D(z, theta):
    """
    Given height and theta (assuming increasing height with index), returns
    the planetary boundary layer height from the Heffter criteria. Assumes
    the data is 1-D with the axis being height. Assumes height in meters, and
    theta in K.
    """
    if z.shape != theta.shape:
        raise ValueError('z and theta must have the same shape')
    if len(z.shape) != 1:  # height axis
        raise ValueError('data has an invalid number of dimensions')
    z = moving_average(z, n=3)
    if not (z < 4000).any():
        raise ValueError('must have data below 4000m')
    theta = moving_average(theta, n=3)
    dtheta = np.diff(theta)
    dz = np.diff(z)
    dtheta_dz = np.zeros_like(dtheta)
    valid = dz != 0
    dtheta_dz[valid] = dtheta[valid]/dz[valid]
    del valid
    inversion = False
    theta_bottom = None
    for i in range(z.shape[0]-1):  # exclude top where dtheta_dz isn't defined
        if z[i] > 4000.:
            # not allowed to have inversion height above 4km
            break
        if inversion:
            # check if we're at PBL top
            if theta[i] - theta_bottom > 2:
                return z[i]
            # check if we're still in an inversion
            if dtheta_dz[i] > 0.005:  # criterion for being in inversion
                pass  # still in inversion, keep going
            else:
                inversion = False
        else:
            if dtheta_dz[i] > 0.005:  # just entered inversion
                theta_bottom = theta[i]
                inversion = True
            else:
                # still not in inversion, keep going
                pass
    # we didn't find a boundary layer height
    # return height of highest dtheta_dz below 4000m
    return z[z < 4000][dtheta_dz[(z < 4000)[:-1]] ==
                       dtheta_dz[(z < 4000)[:-1]].max()][0]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _get_sam_lcl(dataset):
    return zlcl_from_T_RH(dataset['TABS'].values[:, 0], dataset['RELH'].values[:, 0])


def _get_sam_stratocumulus_cbh(dataset):
    cloud_fraction = dataset['CLD'].values
    stratocumulus_base_index = _get_first_index_above_threshold(cloud_fraction, 0.5)
#    stratocumulus_base_index = _get_first_lower_index_below_threshold(
#        cloud_fraction, stratocumulus_height_index, 0.5)
    return_array = np.zeros((cloud_fraction.shape[0],))*np.nan
    found_cbh = stratocumulus_base_index > 0
    return_array[found_cbh] = dataset['z'][stratocumulus_base_index[found_cbh]]
    return return_array


def _get_first_index_above_threshold(array, threshold):
    return _jit_get_first_index_above_threshold(array, threshold).astype(np.int)


@jit(nopython=True)
def _jit_get_first_index_above_threshold(array, threshold):
    number_of_timesteps = array.shape[0]
    number_of_vertical_levels = array.shape[1]
    index_above_threshold = np.zeros((number_of_timesteps,))
    for it in range(number_of_timesteps):
        for iz in range(number_of_vertical_levels):
            if array[it, iz] > threshold:
                index_above_threshold[it] = iz
                break
    return index_above_threshold


def _get_first_lower_index_below_threshold(array, start_index, threshold):
    return _jit_get_first_lower_index_below_threshold(array, start_index, threshold).astype(np.int)


@jit(nopython=True)
def _jit_get_first_lower_index_below_threshold(array, start_index, threshold):
    number_of_timesteps = array.shape[0]
    index_below_threshold = np.zeros((number_of_timesteps,))
    iz_down = 0
    for it in range(number_of_timesteps):
        for iz_down in range(start_index[it]):
            iz = start_index[it] - iz_down
            if (array[it, iz] < threshold) or (iz <= 0):
                index_below_threshold[it] = iz
                break
    return index_below_threshold
