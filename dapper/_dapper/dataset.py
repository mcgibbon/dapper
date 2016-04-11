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
class SoundingDataset(Dataset):

    def __init__(self, filenames, variable_aliases=None):
        if isinstance(filenames, string_types):
            self._sounding_datasets = [xarray.open_dataset(filenames)]
        else:
            self._sounding_datasets = []
            for filename in filenames:
                self._sounding_datasets.append(xarray.open_dataset(filename))
        dataset = construct_sounding_xarray_dataset(self._sounding_datasets)
        super(SoundingDataset, self).__init__(dataset, variable_aliases=variable_aliases)
        self._derive_quantities()

    def _derive_quantities(self):
        self['T'] = (['time', 'iz'], self['tdry'].values + 273.15, {'units': 'degK'})
        self['LCL'] = (['time'], _get_snd_lcl(self), {'units': 'm'})
        self['z_inv'] = (['time'], _get_snd_z_inv(self), {'units': 'm'})
        self['delta_q_bl'] = (['time'], _get_snd_delta_q_bl(self), {'units': 'g/kg'})
        self['stratocumulus_LCL'] = (['time'], _get_snd_stratocumulus_lcl(self), {'units': 'm'})

def construct_sounding_xarray_dataset(sounding_datasets):
    time_axis = [get_xarray_initial_time(ds) for ds in sounding_datasets]
    subset_soundings = [snd.isel(time=slice(0, 1000)) for snd in sounding_datasets]
    for snd in subset_soundings:
        snd.rename({'time':'iz'}, inplace=True)  # time axis is really vertical axis
    dataset = xarray.concat(subset_soundings, 'time')  # new axis for time
    dataset['time'] = (['time'], time_axis)
    return dataset


def theta_from_sounding_dataset(xarray_dataset):
    return (xarray_dataset['tdry'] + 273.15) * (xarray_dataset['pres'].values/(1e3))**2/7.

def get_xarray_initial_time(xarray_dataset):
    return xarray_dataset['time_offset'].values[0]


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
        self['delta_q_bl'] = (['time'], _get_sam_delta_q_bl(self), {'units': 'g/kg'})
        self['stratocumulus_LCL'] = (['time'], _get_sam_stratocumulus_lcl(self), {'units': 'm'})

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


def qv_from_p_T_RH(p, T, RH):
    es = 611.2*np.exp(17.67*(T-273.15)/(T-29.65))
    qvs = 0.622*es/(p-0.378*es)
    rvs = qvs/(1-qvs)
    rv = RH/100. * rvs
    qv = rv/(1+rv)
    return qv


def _get_snd_delta_q_bl(dataset):
    z_inv = dataset['z_inv'].values
    q = qv_from_p_T_RH(dataset['pres'].values*100., dataset['tdry'].values + 273.15,
                       dataset['rh'].values)
    q = q*1e3  # want in g/kg
    z = dataset['alt'].values.astype(np.float64)  # stored as float32
    q_top = get_values_at_heights(q, height_axis=z, height_values=z_inv - 100.)
    q_near_surface = get_values_at_heights(q, z, np.zeros_like(z_inv) + 100.)
    return q_near_surface - q_top


def _get_sam_delta_q_bl(dataset):
    z_inv = dataset['z_inv'].values
    q = dataset['QV'].values
    q_top = get_values_at_heights(q, height_axis=z, height_values=z_inv - 100.)
    q_near_surface = get_values_at_heights(q, z, np.zeros_like(z_inv) + 100.)
    return q_near_surface - q_top


def _get_snd_z_inv(dataset):
    return heffter_pblht(dataset['alt'].values, theta_from_sounding_dataset(dataset.xarray).values)


def _get_sam_z_inv(dataset):
    return heffter_pblht(dataset['z'].values, dataset['THETA'].values)


def get_values_at_heights(array, height_axis, height_values):
    if len(height_values) != array.shape[0]:
        raise ValueError('must have one height for each point in time')
    elif (len(height_axis.shape) == 2) and (height_axis.shape[0] == len(height_values)):
        out_values = []
        for i in range(height_axis.shape[0]):
            value_array = _get_values_at_heights(array[i:i+1, :], height_axis[i, :],
                                                 height_values[i:i+1])
            out_values.append(value_array[0])
        return np.asarray(out_values)
    elif len(height_axis.shape) > 1:
        raise ValueError('height_axis must be 1D or have one axis for each of height_values')
    else:
        return _get_values_at_heights(array, height_axis, height_values)

@jit(nopython=True)
def _get_values_at_heights(array, height_axis, height_values):
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
    return z[:-1][z[:-1] < 4000][np.argmax(dtheta_dz[z[:-1] < 4000])]


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def _get_snd_lcl(dataset):
    return zlcl_from_T_RH(dataset['tdry'].values[:, 0] + 273.15, dataset['rh'].values[:, 0])


def _get_sam_lcl(dataset):
    return zlcl_from_T_RH(dataset['TABS'].values[:, 0], dataset['RELH'].values[:, 0])


def _get_snd_stratocumulus_lcl(dataset):
    z_inv = dataset['z_inv'].values
    T = dataset['T'].values
    RH = dataset['rh'].values
    z = dataset['alt'].values
    T_top = get_values_at_heights(T, height_axis=z, height_values=z_inv - 100.)
    RH_top = get_values_at_heights(RH, height_axis=z, height_values=z_inv - 100.)
    return z_inv + zlcl_from_T_RH(T_top, RH_top)


def _get_sam_stratocumulus_lcl(dataset):
    z_inv = dataset['z_inv'].values
    T = dataset['TABS'].values
    RH = dataset['RELH'].values
    z = dataset['z'].values
    T_top = get_values_at_heights(T, height_axis=z, height_values=z_inv - 100.)
    RH_top = get_values_at_heights(RH, height_axis=z, height_values=z_inv - 100.)
    return z_inv + zlcl_from_T_RH(T_top, RH_top)


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
