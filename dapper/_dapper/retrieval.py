# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:19:59 2016

@author: mcgibbon
"""
from .dataset import Dataset, FilenameDataset, EmptyDataset
from . import export
from . import magic
import numpy as np
import os
import re
import xarray
from datetime import datetime

regex_dataset_dict = {}
regex_dataset_dict.update(magic.regex_dataset_dict)


@export
def get_dataset(data_type, data_source='obs', time_range=None):
    """
    data_type is a string specifying what data is desired (sounding, radiation, etc.)
    """
    data_type = data_type.lower()  # case insensitive searching
    if data_type in regex_dataset_dict:
        return get_regex_dataset(data_type, time_range)
    elif data_type == 'lsf':
        return get_magic_lsf_dataset(time_range)


@export
def get_magic_lsf_dataset(filename_or_leg):
    try:
        leg = magic.get_leg(filename_or_leg)
    except StopIteration:
        leg = filename_or_leg
    lsf_npz_filename = os.path.join(
        '/home/disk/p/mcgibbon/python/magic/run_dir/', leg, 'lsf.npz')
    return get_npz_dataset(lsf_npz_filename)


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
    # we'll assume each coordinate has a different length, so we can find
    # the coordinates of each array based on dimension length
    with np.load(filename) as npz_data:
        data_vars, coords = get_data_vars_and_coords_from_npz(npz_data)
        xarray_dataset = xarray.Dataset(data_vars=data_vars, coords=coords)
    return Dataset(xarray_dataset)


def get_data_vars_and_coords_from_npz(npz_data):
    length_to_coord_dict = get_length_to_coord_dict(npz_data)
    data_vars = get_data_vars_from_npz(npz_data, length_to_coord_dict)
    coords = get_coords_from_npz(npz_data, length_to_coord_dict)
    return data_vars, coords


def get_length_to_coord_dict(npz_data):
    length_to_coord_dict = {}
    coord_names = ('time', 'z')
    for coord in coord_names:
        if coord in npz_data:
            if len(npz_data[coord].shape) > 1:
                raise ValueError('coordinate "{}" is multi-dimensional'.format(coord))
            if len(npz_data[coord]) in length_to_coord_dict:
                raise ValueError('multiple coordinates have the same length')
            length_to_coord_dict[len(npz_data[coord])] = coord
    return length_to_coord_dict


def get_coords_from_npz(npz_data, length_to_coord_dict):
    coords = {}
    for coord in length_to_coord_dict.values():
        coords[coord] = ([coord], npz_data[coord])
    return coords


def get_data_vars_from_npz(npz_data, length_to_coord_dict):
    """
    Given a zipped numpy dataset and a dictionary that maps axis length to coordinate
    names, return the data_vars argument necessary to construct an xarray dataset.
    """
    data_vars = {}
    for key in npz_data.keys():
        key_is_coordinate = key in length_to_coord_dict.values()
        if not key_is_coordinate:
            coords = []
            for length in npz_data[key].shape:
                coords.append(length_to_coord_dict[length])
        data_vars[key] = (coords, npz_data[key][:])
    return data_vars


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
