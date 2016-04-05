# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:38:21 2016

@author: mcgibbon
"""
from . import export
from datetime import timedelta
from datetime import datetime as builtin_datetime
import pytz
import numpy as np


def datetime64_to_builtin_datetime(dt64):
    if isinstance(dt64, np.datetime64):
        return builtin_datetime.utcfromtimestamp(
            (dt64 - np.datetime64('1970-01-01T00:00:00Z')) /
            np.timedelta64(1, 's')).replace(tzinfo=pytz.UTC)
    elif isinstance(dt64, builtin_datetime):
        return dt64
    else:
        raise ValueError('dt64 must be np.datetime64 object')


def cast_to_dapper_datetime(dt):
    if isinstance(dt, np.datetime64):
        dt = datetime64_to_builtin_datetime(dt)
    elif isinstance(dt, datetime):
        return dt
    else:
        return datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second,
                        dt.microsecond, dt.tzinfo)


@export
class datetime(builtin_datetime):

    def as_datetime64(self):
        return np.datetime64(self)

    def __lt__(self, other):
        if isinstance(other, datetime):
            return self < builtin_datetime(other)
        elif isinstance(other, builtin_datetime):
            if builtin_datetime.tzinfo is not None:
                return super(builtin_datetime, self.replace(tzinfo=pytz.UTC)).__lt__(other)
            else:
                return super(builtin_datetime, self.replace(tzinfo=None)).__lt__(other)
        elif isinstance(other, np.datetime64):
            return self.as_datetime64() < other

    def __gt__(self, other):
        if isinstance(other, datetime):
            return self < builtin_datetime(other)
        elif isinstance(other, builtin_datetime):
            if builtin_datetime.tzinfo is not None:
                return super(builtin_datetime, self.replace(tzinfo=pytz.UTC)).__gt__(other)
            else:
                return super(builtin_datetime, self.replace(tzinfo=None)).__gt__(other)
        elif isinstance(other, np.datetime64):
            return self.as_datetime64() > other

    def __eq__(self, other):
        return not self < other and not other < self

    def __ne__(self, other):
        return self < other or other < self

    def __ge__(self, other):
        return not self < other

    def __le__(self, other):
        return not other < self


@export
class TimeRange(object):

    def __init__(self, start_time, end_time):
        self.start = cast_to_dapper_datetime(start_time)
        self.end = cast_to_dapper_datetime(end_time)

    def __iter__(self):
        for item in (self.start, self.end):
            yield item

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError('TimeRange indices must be integers')
        if index == 0:
            return self.start
        elif index == 1:
            return self.end
        else:
            raise IndexError

    def contains(self, time):
        return (self.start < time) and (self.end > time)


def get_netcdf_time(dataset, time_varname='time_offset'):
    try:
        return dataset.variables[time_varname]
    except KeyError:
        return dataset.variables['time']


def correct_day_offsets(time_array):
    """
    Ensures time_array is monotonically increasing, by adding time in increments of
    days to any times that are less than the previous time in the time array, in order.
    Does so in place.

    :param time_array:
    :return:
    """
    for i in range(1, len(time_array)):
        while time_array[i] < time_array[i-1]:
            time_array[i] += timedelta(days=1)
