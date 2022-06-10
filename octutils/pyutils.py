__author__ = "Eligio Maure"
__copyright__ = "Copyright (C) 2022 Eligio Maure"
__license__ = "MIT"

import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from cftime import date2num, num2date
from dateutil.parser import parse
from matplotlib import patches
from matplotlib.path import Path as mPath
from netCDF4 import Dataset
from pyproj import Geod


# ====================
# Time-series analysis
# ====================
class FileError(Exception):
    """A custom exception used to report errors"""

    def __init__(self, message: str):
        super().__init__(message)


class File:
    def __init__(self, file: Path, mode='r'):
        self.path = Dataset(file, mode=mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            exc_info = ''.join(traceback.format_exception(
                exc_type, exc_val, exc_tb))
            raise FileError(exc_info)
        if exc_type is None:
            self.close()

    def read(self, key: str):
        if key == 'time':
            if key in self.path.variables.keys():
                return num2date(self.path[key][:],
                                units=self.path[key].units)
            return parse(self.path.getncattr('time_coverage_start'))
        return self.path[key][:]

    def get_attr(self, name, loc='/'):
        if loc == '/':
            if 'time' in name:
                return parse(self.path.getncattr(name))
            return self.path.getncattr(name)
        return self.path[loc].getncattr(name)

    def get_mean_date(self):
        start = self.get_attr(name='time_coverage_start')
        end = self.get_attr(name='time_coverage_end')
        return mean_date(dates=[start, end])

    def close(self):
        self.path.close()
        [setattr(self, key, None)
         for key in self.__dict__.keys()]


def mean_date(dates: list, units: str = None, calendar: str = None):
    """
    Returns the mean date from a list of dates
    Parameters
    ----------
    dates: list
        list of datetime to be converted into seconds since epoch
    units: str
        epoch to be used in converting the dates
    calendar: str

    Returns
    -------
    date: datetime
        the mean date from dates
    """
    if units is None:
        units = 'seconds since 1970-01-01 00:00:00'
    if calendar is None:
        calendar = 'gregorian'
    return secs2date(
        times=np.array(
            date2secs(dates=dates,
                      units=units,
                      calendar=calendar)
        ).mean(),
        units=units,
        calendar=calendar
    )


def date2secs(dates, units: str = None, calendar: str = None):
    """
    Converts datetime list to seconds since epoch
    Parameters
    ----------
    dates: datetime | list
        list of datetime to be converted to seconds since epoch
    units: str
        epoch to be used in converting the dates
    calendar: str

    Returns
    -------
    times: list
        list of times or dates in seconds since epoch
    """
    if units is None:
        units = 'seconds since 1970-01-01 00:00:00'
    if calendar is None:
        calendar = 'gregorian'
    return date2num(dates=dates, units=units, calendar=calendar)


def secs2date(times, units: str = None, calendar: str = None):
    """
    Converts times list into dates since epoch
    Parameters
    ----------
    times: int | list | np.array
        list of times or seconds to be converted into dates since epoch
    units: str
        epoch to be used in converting the dates
    calendar: str

    Returns
    -------
    dates: list
        list of dates since epoch
    """
    if units is None:
        units = 'seconds since 1970-01-01 00:00:00'
    if calendar is None:
        calendar = 'gregorian'
    return num2date(times=times, units=units, calendar=calendar)


def add_marker(ax, bbox, marker=None, size=200, fc=None):
    if marker is None:
        marker = 'o'
    for x, y in zip(bbox['lon'], bbox['lat']):
        if type(x) == list:
            path = get_path(bbox={'lon': x, 'lat': y})
            patch = patches.PathPatch(path, lw=2)
            ax.add_patch(patch)
        else:
            if fc is None:
                ax.scatter(x, y, s=size, edgecolors='k', marker=marker)
            else:
                ax.scatter(x, y, s=size, edgecolors='k', marker=marker, c=fc)
    return


def get_path(bbox: dict):
    x0, x1 = bbox['lon']
    y0, y1 = bbox['lat']

    # path vertex coordinates
    vertices = [
        (x0, y0),  # left, bottom
        (x0, y1),  # left, top
        (x1, y1),  # right, top
        (x1, y0),  # right, bottom
        (x0, y0),  # ignored
    ]

    codes = [
        mPath.MOVETO,
        mPath.LINETO,
        mPath.LINETO,
        mPath.LINETO,
        mPath.CLOSEPOLY,
    ]

    return mPath(vertices, codes)


def area_mask(bbox: dict, lon, lat):
    path = get_path(bbox=bbox)

    # create a mesh grid for the whole image
    if len(lon.shape) == 1:
        x, y = np.meshgrid(lon, lat)
    else:
        x, y = lon, lat
    # mesh grid to a list of points
    points = np.vstack((x.ravel(), y.ravel())).T

    # select points included in the path
    mask = path.contains_points(points)
    return np.array(mask).reshape(x.shape)


def harv_dist(px: float, py: float, lon, lat, datum: str = 'WGS84'):
    """
    Distance from a point x, y in geographical space
    """
    p = np.pi / 180.
    g = Geod(ellps=datum)

    a = (g.a ** 2 * np.cos(py * p)) ** 2
    b = (g.b ** 2 * np.sin(py * p)) ** 2

    c = (a * np.cos(py * p)) ** 2
    d = (b * np.cos(py * p)) ** 2

    r = np.sqrt((a + b) / (c + d))

    if len(lon.shape) == 1:
        x, y = np.meshgrid(lon, lat)
    else:
        x, y = lon, lat

    px = np.ones(x.shape) * px
    py = np.ones(x.shape) * py
    e = 0.5 - np.cos((y - py) * p) / 2 + np.cos(
        y * p) * np.cos(py * p) * (
                1 - np.cos((px - x) * p)) / 2
    # 2*R*asin..
    return 2 * r * np.arcsin(e ** .5)


def pixel_extract(sds, mask, scale):
    """
    :param sds:
    :type sds: np.array
    :param mask:
    :type mask: np.array
    :param scale:
    :type scale: str
    :return:
    :rtype: list
    """
    masked = sds[mask]
    valid = np.ma.compressed(masked)
    valid_px = valid.size
    total_px = masked.size
    invalid_px = total_px - valid_px

    'pixel_count, valid, invalid, min, max, mean, median, std, pixel_value'
    if np.all(masked.mask):
        return [total_px, valid_px, invalid_px, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    if scale == 'log':
        sds_mean = 10 ** np.log10(valid).mean()
        sds_std = 10 ** np.log10(valid).std()
    else:
        sds_mean = valid.mean()
        sds_std = valid.std()
    sds_max = valid.max()
    sds_min = valid.min()
    sds_med = np.median(valid)
    dset = [total_px, valid_px, invalid_px, sds_min, sds_max, sds_mean, sds_med, sds_std]
    fmt = ['g', 'g', 'g', '.6f', '.6f', '.6f', '.6f', '.6f', '.6f']
    out = []
    for v, f in zip(dset, fmt):
        if v == sds.fill_value:
            out.append('')
            continue
        out.append(f'{v:{f}}')
    return out


def pyextract(var, bbox, ifiles, ofile, window=None, mode='w'):
    """
    Extract point of region from a series of satellite images

    :param var: variable name being extracted from the file-list
    :type var: str
    :param bbox: point(s)/area(s) of interest.
                 In case of point the bbox should is formatted like
                 {lon: [lon], lat: [value]} for a single point or
                 {lon: [lon0, lon1, ..., n], lat: [lat0, lat1, ..., n]} for more two or more points.
                 In case of region the bbox should be formatted such that
                 {lon: [[min, max]], lat: [[min, max]]} for a single region or
                 {lon: [[min0, max0], [min1, max1],...], lat: [[min0, max0], [min1, max1],...]} for more than one region
    :type bbox: dict
    :param ifiles: list of files to use for data extract
    :type ifiles: list
    :param ofile: fullpath of output text file
    :type ofile: Path
    :param window: pixel window size of the satellite data extract made around the point, optional: default = 3
    :type window: int
    :param mode: file open mode, w - write mode, a - append mode
    :type mode: str
    :return: pandas dataframe with extract data
    :rtype: pd.DataFrame
    """
    if window is None:
        window = 3
    if window % 2 == 0:
        raise Exception('Window must be an odd number!!')

    with open(ofile, mode=mode) as txt:
        if mode == 'w':
            txt.writelines('file,lon,lat,variable,'
                           'time_start,time_end,'
                           'pixel_count,valid,invalid,'
                           'min,max,mean,median,std,pixel_value\n')
        scale = 'log' if 'chl' in var.lower() else 'lin'
        for f in ifiles:
            with File(file=f) as fid:
                lat = fid.read(key='lat')
                lon = fid.read(key='lon')
                sds = fid.read(key=var)
                start = fid.get_attr(name='time_coverage_start').strftime('%FT%H:%M:%SZ')
                end = fid.get_attr(name='time_coverage_end').strftime('%FT%H:%M:%SZ')

            for px, py in zip(bbox['lon'], bbox['lat']):
                if type(px) == list:
                    mask = area_mask(bbox={'lon': px, 'lat': py}, lon=lon, lat=lat)
                    px = ';'.join(f'{v}' for v in px)
                    py = ';'.join(f'{v}' for v in py)
                    pxv = ''
                else:
                    dist = harv_dist(px=px, py=py, lon=lon, lat=lat)
                    center = np.where(dist == dist.min())
                    mask = np.zeros_like(dist)
                    mask[center] = 1
                    kernel = np.ones((window, window), np.uint8)
                    mask = np.bool_(cv2.dilate(mask, kernel, iterations=1))
                    pxv = sds[center][0]
                    pxv = f'{pxv:.6f}' if pxv != sds.fill_value else ''

                line = pixel_extract(sds=sds, mask=mask, scale=scale)
                joined = ','.join([f.name, f'{px}', f'{py}', var, start, end] + line + [pxv])
                txt.writelines(f'{joined}\n')
    return pd.read_csv(ofile)


def preallocate(file, n):
    """
    Creates an empty array to hold timeseries data given shape and type.
    Pre-allocation increases speed and avoids resizing of output array on demand.
    :param file: file from which to get the shape (lat, lon)
    :type file: Path
    :param n: the length of the timeseries. Often, the number of images to load.
    :type n: int
    :return: lon, lat, and pre-allocated array
    :rtype: tuple
    """
    with File(file=file, mode='r') as fid:
        lat = fid.read(key='lat')
        lon = fid.read(key='lon')
    sds = np.ma.empty((n, lat.size, lon.size), dtype=np.float32)
    return lon, lat, sds


def get_timeseries(files, var):
    """
    Return a timeseries from a collection of images.
    :param files: input list of files from which to get the timeseries.
    :type files: list
    :param var: variable name in the file list.
    :type var: str
    :return: tuple of lon, lat, timeseries data
    :rtype: tuple
    """
    dates = []
    append = dates.append
    lon, lat, sds = preallocate(file=files[0], n=len(files))

    for j, f in enumerate(files):
        with File(file=f, mode='r') as fid:
            dt = fid.read(key=var)
            sds[j, :, :] = dt
            append(fid.get_mean_date())
    fill_value = dt.fill_value
    np.ma.set_fill_value(sds, fill_value=fill_value)
    return lon, lat, sds, dates


def get_composite(files, var):
    """
    Return the composite of images created from a list of input files.
    :param files: input list of files from which to get the composite.
    :type files: list
    :param var: variable name in the file list.
    :type var: str
    :return: tuple of lon, lat, composite data
    :rtype: tuple
    """
    scale = 'log' if 'chl' in var.lower() else 'lin'
    lon, lat, sds, _ = get_timeseries(files=files, var=var)
    if scale == 'log':
        return lon, lat, np.ma.power(10, np.ma.log10(sds).mean(axis=0))
    return lon, lat, sds.mean(axis=0)


def annual_max(files, var):
    """
    Return the maximum and the time when it occurs from the input files.
    :param files: input list of files from which the maximum is obtained.
    :type files: list
    :param var: variable name in the list of files.
    :type var: str
    :return: tuple of lon, lat, data, and time of max
    :rtype: tuple
    """
    lon, lat, sds, dates = get_timeseries(files=files, var=var)
    fill_value = sds.fill_value

    # Max data
    mask = sds.mean(axis=0).mask
    data = np.ma.amax(sds, axis=0)
    data = np.ma.masked_where(mask, data)
    np.ma.set_fill_value(data, fill_value=fill_value)

    idx = np.ma.argmax(sds, axis=0)
    doy = np.array([d.timetuple().tm_yday for d in dates])[idx]
    doy = np.ma.masked_where(idx == 0, doy)
    return lon, lat, data, doy


def annual_min(files, var):
    """
    Return the minimum and the time when it occurs from the input files.
    :param files: input list of files from which the minimum is obtained.
    :type files: list
    :param var: variable name in the list of files.
    :type var: str
    :return: tuple of lon, lat, data, and time of min
    :rtype: tuple
    """
    lon, lat, sds, dates = get_timeseries(files=files, var=var)
    fill_value = sds.fill_value

    # Min data
    mask = sds.mean(axis=0).mask
    data = np.ma.amin(sds, axis=0)
    data = np.ma.masked_where(mask, data)
    np.ma.set_fill_value(data, fill_value=fill_value)

    idx = np.ma.argmin(sds, axis=0)
    doy = np.array([d.timetuple().tm_yday for d in dates])[idx]
    doy = np.ma.masked_where(idx == 0, doy)
    return lon, lat, data, doy
