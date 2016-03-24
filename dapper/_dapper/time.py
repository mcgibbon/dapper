# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:38:21 2016

@author: mcgibbon
"""
from . import export
from datetime import timedelta


class TimeRange(object):

    def __init__(self, start_time, end_time):
        self.start = start_time
        self.end = end_time

    def __iter__(self):
        for item in (self.start, self.end):
            yield item

    def contains(self, time):
        return (time > self.start) and (time < self.end)


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
