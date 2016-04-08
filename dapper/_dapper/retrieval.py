# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:19:59 2016

@author: mcgibbon
"""
from .dataset import Dataset, FilenameDataset, EmptyDataset
from . import export
from . import magic
from .util import zlcl_from_T_RH
import numpy as np
import os
import re
import xarray
from datetime import datetime

regex_dataset_dict = {}
regex_dataset_dict.update(magic.regex_dataset_dict)
coordinate_names = ('time', 'height')


@export
def get_dataset(data_type, time_range=None, **kwargs):
    """
    data_type is a string specifying what data is desired (sounding, radiation, etc.)
    """
    data_type = data_type.lower()  # case insensitive searching
    if data_type in regex_dataset_dict:
        return get_regex_dataset(data_type, time_range, **kwargs)
    elif data_type == 'cloud_fraction':
        return get_cloud_fraction_dataset(time_range, **kwargs)
    elif data_type == 'mwr':
        return get_mwr_dataset(time_range, **kwargs)
    elif data_type == 'decoupling':
        return get_decoupling_dataset(time_range, **kwargs)
    else:
        raise ValueError('Unable to get dataset for data_type "{}"'.format(data_type))


@export
def get_magic_forcing_dataset(filename_or_leg, forcing_type='lsf'):
    try:
        leg = magic.get_leg(filename_or_leg)
    except StopIteration:
        leg = filename_or_leg
    lsf_npz_filename = os.path.join(
        '/home/disk/p/mcgibbon/python/magic/run_dir/', leg, '{}.npz'.format(forcing_type))
    return get_npz_dataset(lsf_npz_filename)


def get_decoupling_dataset(time_range=None, window='3H'):
    ceilometer = get_dataset('ceilometer', time_range=time_range)
    first_cbh = ceilometer['first_cbh']
    first_cbh = first_cbh[first_cbh.values < 3000.].resample(
        window, dim='time', how=lambda x, axis=None: np.percentile(x, q=95, axis=axis))
    cf_dataset = get_dataset('cloud_fraction', time_range=time_range, window=window, label='left')
    cloud_fraction, first_cbh = xarray.align(cf_dataset['low_cloud_fraction'], first_cbh)
    first_cbh[cloud_fraction < 0.5] = np.nan
    surface = get_dataset('marmet', time_range=time_range).xarray.resample(
        window, dim='time', how='mean')
    surface, first_cbh = xarray.align(surface, first_cbh, join='outer')
    zlcl = zlcl_from_T_RH(surface['air_temperature'], surface['relative_humidity'])
    data_vars = {
        'LCL': (['time'], zlcl, {'units': 'm'}),
        'cbh': (['time'], first_cbh, {'units': 'm'}),
    }
    coords = {'time': (['time'], surface['time'])}
    return Dataset(xarray.Dataset(data_vars, coords))


def get_cloud_fraction_dataset(time_range=None, window='3H', label='center'):
    ceil_dataset = get_dataset('ceilometer', time_range=time_range)
    low_cld_frac, cld_frac = get_cloud_fraction_arrays_from_ceil_dataset(ceil_dataset)
    ceil_dataset['cloud_fraction'] = (('time',), cld_frac)
    ceil_dataset['low_cloud_fraction'] = (('time',), low_cld_frac)
    return ceil_dataset.resample(window, label=label)


def get_cloud_fraction_arrays_from_ceil_dataset(ceil_dataset):
    cloud_fraction_array = (~np.isnan(ceil_dataset['first_cbh'].values)).astype(np.float64)
    low_cloud_fraction_array = np.zeros_like(cloud_fraction_array)
    low_cloud_fraction_array[:] = cloud_fraction_array[:]
    low_cloud_fraction_array[ceil_dataset['first_cbh'].values > 3000.] = 0.
    return low_cloud_fraction_array, cloud_fraction_array


def get_mwr_dataset(time_range=None, apply_qc_mask=True):
    xarray_datasets = []
    for filename in get_mwr_filenames(time_range):
        xarray_datasets.append(get_mwr_xarray_dataset_from_filename(
            filename, apply_qc_mask=apply_qc_mask))
    return Dataset(xarray.concat(xarray_datasets, dim='time'))


def get_mwr_filenames(time_range):
    mwr_dir = '/home/disk/eos4/mcgibbon/nobackup/MAGIC/all/new_mwr_retrieval/'
    if time_range is None:
        filenames = get_all_dat_in_directory(mwr_dir)
    else:
        year_month_pairs = get_year_month_pairs(time_range)
        filenames = []
        for year, month in year_month_pairs:
            filenames.append(
                os.path.join(mwr_dir, 'MAG{:04d}{:02d}_retrieval.dat'.format(year, month)))
    return filenames


def get_mwr_xarray_dataset_from_filename(filename, apply_qc_mask=True):
    data = np.genfromtxt(
        filename,
        usecols=range(0, 13) + range(20, 23),
        names=('year', 'month', 'day', 'hour', 'minute', 'second', 'day_in_year',
               'pwv', 'pwv_error', 'lwp', 'lwp_error', 'firstguesspwv',
               'firstguesslwp', 'converged', 'measurement_rmse',
               'measurement_vs_model_rmse')
    )
    datetimes = np.asarray(get_datetimes_from_mwr_data(data))
    if apply_qc_mask:
        valid = data['converged'][:].astype(np.bool) & (data['measurement_vs_model_rmse'] < 1.)
    else:
        valid = np.ones_like(data['converged'][:]).astype(np.bool)
    xarray_dataset = xarray.Dataset(
        data_vars={
            'pwv': (['time'], data['pwv'][valid], {'units': 'cm'}),
            'pwv_error': (['time'], data['pwv_error'][valid], {'units': 'cm'}),
            'lwp': (['time'], data['lwp'][valid], {'units': 'mm'}),
            'lwp_error': (['time'], data['lwp_error'][valid], {'units': 'mm'}),
            'converged': (['time'], data['converged'][valid].astype(np.bool)),
            'measurement_vs_model_rmse': (
                ['time'], data['measurement_vs_model_rmse'][valid], {'units': 'K'}),
        },
        coords={
            'time': (['time'], datetimes[valid]),
        }
    )
    return xarray_dataset


def get_datetimes_from_mwr_data(data):
    datetimes = np.zeros_like(data['year'], dtype=datetime)
    datetime_args = zip(data['year'][:].astype(np.int), data['month'][:].astype(np.int),
                        data['day'][:].astype(np.int), data['hour'][:].astype(np.int),
                        data['minute'][:].astype(np.int), data['second'][:].astype(np.int))
    for i in range(len(datetimes)):
        datetimes[i] = datetime(*datetime_args[i])
    return datetimes


def get_year_month_pairs(time_range):
    pairs = []
    start, end = time_range.start, time_range.end
    for year in range(start.year + 1, end.year):
        # add all months for intermediate years, if any
        pairs.extend([(year, month) for month in range(1, 13)])
    if end.year > start.year:
        pairs.extend([(start.year, month) for month in range(start.month, 13)])
        pairs.extend([(end.year, month) for month in range(1, end.month + 1)])
    elif end.year == start.year:
        pairs.extend([(start.year, month) for month in range(start.month, end.month + 1)])
    return sorted(pairs, key=lambda pair: pair[0] + pair[1]/24.)


def get_all_dat_in_directory(directory):
    filenames = os.listdir(directory)
    return [os.path.join(directory, name) for name in filenames
            if (len(name) > 4) and (name[:-4] == '.dat')]


def get_regex_dataset(data_type, time_range=None):
    directory = regex_dataset_dict[data_type]['directory']
    regex_string = regex_dataset_dict[data_type]['regex']
    filenames = get_filenames(directory, regex_string, time_range)
    if len(filenames) == 0:
        return EmptyDataset()
    else:
        aliases = regex_dataset_dict[data_type]['aliases']
        return FilenameDataset(filenames, variable_aliases=aliases)


def get_npz_dataset(filename):
    """
    Load data from the given npz filename as a Dataset. Assume that 1D variables
    are in time, and 2D are in time-height.
    """
    with np.load(filename) as npz_data:
        data_vars, coords = get_data_vars_and_coords_from_npz(npz_data)
        xarray_dataset = xarray.Dataset(data_vars=data_vars, coords=coords)
    return Dataset(xarray_dataset)


def get_data_vars_and_coords_from_npz(npz_data):
    data_vars = get_data_vars_from_npz(npz_data)
    coords = get_coords_from_npz(npz_data)
    return data_vars, coords


def get_coords_from_npz(npz_data):
    """
    Returns a coords dictionary required to make an xarray dataset.
    """
    coords = {}
    for coord in coordinate_names:
        if coord in npz_data:
            coord_data = get_coord_data(npz_data, coord)
            coords[coord] = ([coord], coord_data)
    return coords


def get_coord_data(npz_data, coord):
    if len(npz_data[coord].shape) > 1:
        return np.arange(npz_data[coord].shape[coordinate_names.index(coord)])
    else:
        return npz_data[coord][:]


def get_data_vars_from_npz(npz_data):
    """
    Given a zipped numpy dataset and a dictionary that maps axis length to coordinate
    names, return the data_vars argument necessary to construct an xarray dataset.
    Assume that variable axes are given by coordinate_names[:dimensionality]
    """
    data_vars = {}
    for key in npz_data.keys():
        if key not in coordinate_names:
            coords = get_coords_for_array(npz_data[key])
            data_vars[key] = (coords, npz_data[key][:])
    return data_vars


def get_coords_for_array(array):
    """
    Returns the xarray coordinate list for a given numpy array, assuming variable axes
    are given by coordinate_names[:dimensionality].
    """
    dimensionality = len(array.shape)
    if dimensionality > len(coordinate_names):
        raise ValueError(
            'Not sure what to assume about axes for {} dimensional data'.format(
                dimensionality))
    else:
        return coordinate_names[:dimensionality]


def get_filenames(foldername, regex_str, time_range=None):
    regex_prog = re.compile(regex_str)
    filenames = os.listdir(foldername)
    return_list = []
    for filename in filenames:
        if filename_matches_criteria(filename, regex_prog, time_range):
            return_list.append(os.path.join(foldername, filename))
    return sorted(return_list)


def filename_matches_criteria(filename, regex_prog, time_range=None):
    match = regex_prog.match(filename)
    if match is None:
        return False
    elif time_range is None:
        return True
    elif time_range.contains(get_datetime_from_match(match)):
        return True
    else:
        return False


def get_datetime_from_match(match):
    # groups of regex program should be constructed to extract the file start time
    return datetime(*[int(arg) for arg in match.groups()])
