# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:33:45 2016

@author: mcgibbon
"""
from . import export
from .time import TimeRange, datetime
import os
import re
import pytz

root_dir = '/home/disk/eos4/mcgibbon/nobackup/MAGIC/all/'
regex_dataset_dict = {
    'soundings': {
        'regex': r'magsondewnpnM1\.b1\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.custom\.cdf',
        'directory': os.path.join(root_dir, 'snd'),
        'aliases': {},
    },
    'ceilometer': {
        'regex': r'magceilM1\.b1\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.custom\.nc',
        'directory': os.path.join(root_dir, 'ceil'),
        'aliases': {},
    },
    'old_mwr_retrieval': {
        'regex': r'magmwrret1liljclouM1\.s1\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.cdf',
        'directory': os.path.join(root_dir, 'mwrret'),
        'aliases': {},
    },
    'surface_radiation': {
        'regex': r'magprpradM1\.a1\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.custom\.cdf',
        'directory': os.path.join(root_dir, 'prprad'),
        'aliases': {},
    },
    'wacr': {
        'regex': r'magmwacrM1\.a1\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.custom\.cdf',
        'directory': os.path.join(root_dir, 'wacr'),
        'aliases': {},
    },
    'parsivel_disdrometer': {
        'regex': r'magpars2S1\.b1\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.cdf',
        'directory': os.path.join(root_dir, 'pars'),
        'aliases': {},
    },
    'kazr_precipitation': {
        'regex': r'KAZR_precipitation.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d).nc',
        'directory': os.path.join(root_dir, 'kazr_precip_5min'),
        'aliases': {},
    },
    'marmet': {
        'regex': r'magmarmet\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)(\d\d)\.custom\.cdf',
        'directory': os.path.join(root_dir, 'marmet'),
        'aliases': {},
    },
}


def get_leg(sam_filename):
    prog = re.compile(
        r'MAG(?:2|3)D\.(\d+A|\d+B)\.(\d{4})(\d\d)(\d\d)\.(\d\d)(\d\d)_\d+h_'
        r'(?:.+_)?\d+x\d+x\d+_LES.nc')
    return prog.finditer(sam_filename).next().group(1)


def get_leg_times(filename):
    leg_times = {}
    prog = re.compile(
        r'(\d{2}):     (\d{4}) (\d\d) (\d\d) (\d\d) (\d\d) (\d\d)(?: |~)'
        r'(\d+\.\d+)   (\d{4}) (\d\d) (\d\d) (\d\d) (\d\d) (\d\d)(?: |~)'
        r'(\d+\.\d+)   (\d{4}) (\d\d) (\d\d) (\d\d) (\d\d) (\d\d)(?: |~)'
        r'(\d+\.\d+)   (\d{4}) (\d\d) (\d\d) (\d\d) (\d\d) (\d\d)(?: |~)'
        r'(\d+\.\d+)')
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        m = prog.match(line)
        if m is not None:
            legno = int(m.group(1))
            leg_times[str(legno) + 'A'] = TimeRange(
                datetime(
                    *[int(item) for item in m.groups()[1:7]]).replace(
                        tzinfo=pytz.UTC),
                datetime(
                    *[int(item) for item in m.groups()[8:14]]).replace(
                        tzinfo=pytz.UTC)
            )
            leg_times[str(legno) + 'B'] = TimeRange(
                datetime(
                    *[int(item) for item in m.groups()[15:21]]).replace(
                        tzinfo=pytz.UTC),
                datetime(
                    *[int(item) for item in m.groups()[22:28]]).replace(
                        tzinfo=pytz.UTC)
            )
    return leg_times


leg_times_filename = os.path.join(root_dir, 'isar', 'MagicLegTimes.txt')
leg_times = get_leg_times(leg_times_filename)
export(leg_times, as_name='magic_leg_times')
