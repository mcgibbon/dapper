# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:19:59 2016

@author: mcgibbon
"""
#from .dataset import Dataset, FilenameDataset, EmptyDataset, SoundingDataset
from .dataset import *
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
    elif data_type in ('sounding', 'soundings'):
        return get_regex_dataset(
            'soundings', time_range, dataset_func=get_sounding_dataset_from_filenames, **kwargs)
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


def get_dataset_from_filenames(filenames, variable_aliases=None):
    return Dataset(get_xarray_datasets_from_filenames(filenames), variable_aliases=variable_aliases)


def get_sounding_dataset_from_filenames(filenames, variable_aliases=None):
    sounding_datasets = get_xarray_datasets_from_filenames(filenames)
    dataset = construct_sounding_xarray_dataset(sounding_datasets)
    dataset['T'] = (['time', 'iz'], dataset['tdry'].values + 273.15, {'units': 'degK'})
    dataset['LCL'] = (['time'], _get_snd_lcl(dataset), {'units': 'm'})
    dataset['z_inv'] = (['time'], _get_snd_z_inv(dataset), {'units': 'm'})
    dataset['delta_q_bl'] = (['time'], _get_snd_delta_q_bl(dataset), {'units': 'g/kg'})
    dataset['stratocumulus_LCL'] = (['time'], _get_snd_stratocumulus_lcl(dataset), {'units': 'm'})
    return Dataset(dataset, variable_aliases=variable_aliases)


def get_xarray_datasets_from_filenames(filenames):
    if isinstance(filenames, string_types):
        datasets = [xarray.open_dataset(filenames)]
    else:
        datasets = []
        for filename in filenames:
            datasets.append(xarray.open_dataset(filename))
    return datasets


def construct_sounding_xarray_dataset(sounding_datasets):
    time_axis = [get_xarray_initial_time(ds) for ds in sounding_datasets]
    subset_soundings = [snd.isel(time=slice(0, 1000)) for snd in sounding_datasets]
    for snd in subset_soundings:
        snd.rename({'time':'iz'}, inplace=True)  # time axis is really vertical axis
    dataset = xarray.concat(subset_soundings, 'time')  # new axis for time
    dataset['time'] = (['time'], time_axis)
    return dataset


def get_regex_dataset(data_type, time_range=None, dataset_func=get_dataset_from_filenames):
    directory = regex_dataset_dict[data_type]['directory']
    regex_string = regex_dataset_dict[data_type]['regex']
    filenames = get_filenames(directory, regex_string, time_range)
    if len(filenames) == 0:
        return EmptyDataset()
    else:
        aliases = regex_dataset_dict[data_type]['aliases']
        return dataset_func(filenames, variable_aliases=aliases)


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


def theta_from_sounding_dataset(xarray_dataset):
    return (xarray_dataset['tdry'] + 273.15) * (xarray_dataset['pres'].values/(1e3))**2/7.


def get_xarray_initial_time(xarray_dataset):
    return xarray_dataset['time_offset'].values[0]


@export
def load_sam_dataset(filename_or_filenames):
    dataset = get_xarray_dataset_from_filename_or_filenames(filename_or_filenames)
    if isinstance(filename_or_filenames, string_types):
        year = get_sam_year_from_filename(filename_or_filenames)
    else:
        year = get_sam_year_from_filename(filename_or_filenames[0])
    dataset['time'] = (['time'], day_in_year_to_datetime(dataset['time'], year))
    dataset['stratocumulus_cbh'] = (['time'], _get_sam_stratocumulus_cbh(dataset), {'units': 'm'})
    dataset['LCL'] = (['time'], _get_sam_lcl(dataset), {'units': 'm'})
    dataset['z_inv'] = (['time'], _get_sam_z_inv(dataset), {'units': 'm'})
    dataset['delta_q_bl'] = (['time'], _get_sam_delta_q_bl(dataset), {'units': 'g/kg'})
    dataset['stratocumulus_LCL'] = (['time'], _get_sam_stratocumulus_lcl(dataset), {'units': 'm'})
    return Dataset(dataset)


def get_xarray_dataset_from_filename_or_filenames(f):
    if isinstance(f, string_types):
        return xarray.open_dataset(f)
    else:
        return xarray.open_mfdataset(f)


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
    z = dataset['z'].values
    q = dataset['QV'].values
    q_top = get_values_at_heights(q, height_axis=z, height_values=z_inv - 100.)
    q_near_surface = get_values_at_heights(q, z, np.zeros_like(z_inv) + 100.)
    return q_near_surface - q_top


def _get_snd_z_inv(dataset):
    return heffter_pblht(dataset['alt'].values, theta_from_sounding_dataset(dataset).values)


def _get_sam_z_inv(dataset):
    return heffter_pblht(dataset['z'].values, dataset['THETA'].values)


def qv_from_p_T_RH(p, T, RH):
    es = 611.2*np.exp(17.67*(T-273.15)/(T-29.65))
    qvs = 0.622*es/(p-0.378*es)
    rvs = qvs/(1-qvs)
    rv = RH/100. * rvs
    qv = rv/(1+rv)
    return qv


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
