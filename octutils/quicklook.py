import re
import textwrap
import time
import traceback
from decimal import Decimal
from pathlib import Path

import cartopy.crs as ccrs
import h5py
import numpy as np
import pyproj
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.ticker import MaxNLocator
from netCDF4 import Dataset
from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.kd_tree import resample_nearest
from termcolor import colored


def interp2d(array: np.array, interval: int):
    """
        Bilinear interpolation of SGLI geo-location corners to a spatial grid

        Parameters
        ----------
        array: np.array
            either lon or lat
        interval: int
            resampling interval in pixels

        Return
        ------
        out_geo: np.array
            2-D array with dims == to geophysical variables
    """
    sds = np.concatenate((array, array[-1].reshape(1, -1)), axis=0)
    sds = np.concatenate((sds, sds[:, -1].reshape(-1, 1)), axis=1)

    ratio_0 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
        (sds.shape[0] * interval, sds.shape[1] - 1))

    ratio_1 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32).reshape(-1, 1),
        (sds.shape[0] - 1, (sds.shape[1] - 1) * interval))

    sds = np.repeat(sds, interval, axis=0)
    sds = np.repeat(sds, interval, axis=1)
    interp = (1. - ratio_0) * sds[:, :-interval] + ratio_0 * sds[:, interval:]
    return (1. - ratio_1) * interp[:-interval, :] + ratio_1 * interp[interval:, :]


class FileError(Exception):
    """A custom exception used to report errors"""

    def __init__(self, message: str):
        super().__init__(message)


class File:
    def __init__(self, file: Path, mode: str = 'r', **kw):
        """
        Interface for data I/O (JAXA's hdf5 and NASA's netcdf4)
        :param file:
        :type file:
        :param mode:
        :type mode:
        :param kw:
        :type kw:
        """
        self.var = None
        self.obj = None
        self.file = file
        self.fill_value = -32767
        self.area_id = 'custom'
        self.description = 'ECOP training'
        self.proj_name = 'laea'
        self.datum = 'WGS84'
        if kw is not None:
            for key, val in kw.items():
                setattr(self, key, val)
        self.proj = None
        if file.suffix == '.nc':
            self.obj = Dataset(file, mode=mode)
            self.path = self.obj.groups['geophysical_data']
            self.lon = self.obj.groups['navigation_data']['longitude'][:]
            self.lat = self.obj.groups['navigation_data']['latitude'][:]
            self.glob_attrs = {
                at: self.obj.getncattr(at) for at in self.obj.ncattrs()}

        if file.suffix == '.h5':
            self.obj = h5py.File(file, mode=mode)
            self.path = self.obj['Image_data']
            self.glob_attrs = self.fmt_attr(path='Global_attributes')
            self.lon = self.get_geo(key='Longitude')
            self.lat = self.get_geo(key='Latitude')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            exc_info = ''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback))
            raise FileError(exc_info)
        if exc_type is None:
            self.close()

    # -----------
    # Get methods
    # -----------

    def get_data(self, key: str = None):
        if key is None:
            key = self.var
        if key is None:
            raise KeyError
        if self.file.suffix == '.nc':
            return self.path[key][:]
        if self.file.suffix == '.h5':
            return self.get_sds(key=key)
        raise FileError('Unexpected file format')

    def get_dn(self, key: str):
        path = self.obj[f'Geometry_data'] \
            if key[:3].lower() in ('lon', 'lat') else \
            self.path
        return path[key][:]

    def get_sds(self, key: str):
        """
            https://shikisai.jaxa.jp/faq/docs/GCOM-C_Products_Users_Guide_entrylevel__attach4_jp_191007.pdf#page=46
        """
        attrs = self.h5_attrs(key=key).pop(key)
        sdn = self.get_dn(key=key)

        if key == 'QA_flag':
            return np.ma.squeeze(sdn).astype(np.int32)

        mask = False
        if 'Error_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Error_DN')), True, False)
        if 'Land_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Land_DN')), True, False)
        if 'Cloud_error_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Cloud_error_DN')), True, False)
        if 'Retrieval_error_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Retrieval_error_DN')), True, False)
        if ('Minimum_valid_DN' in attrs.keys()) and ('Maximum_valid_DN' in attrs.keys()):
            mask = mask | np.where((sdn <= attrs.pop('Minimum_valid_DN')) |
                                   (sdn >= attrs.pop('Maximum_valid_DN')), True, False)

        # Convert DN to PV
        slope, offset = 1, 0
        if 'NWLR' in key:
            if ('Rrs_slope' in attrs.keys()) and \
                    ('Rrs_slope' in attrs.keys()):
                slope = attrs.pop('Rrs_slope')
                offset = attrs.pop('Rrs_offset')
        else:
            if ('Slope' in attrs.keys()) and \
                    ('Offset' in attrs.keys()):
                slope = attrs.pop('Slope')
                offset = attrs.pop('Offset')

        sds = np.ma.masked_where(
            mask, sdn * slope + offset
        ).astype(np.float32)
        np.ma.set_fill_value(sds, self.fill_value)
        return sds

    def get_geo(self, key: str):
        if self.file.suffix == '.nc':
            return getattr(self, key[:3].lower())
        data = self.get_sds(key=key)

        interval = self.fmt_attr(
            path=f'Geometry_data/{key}'
        ).get('Resampling_interval')

        attrs = self.fmt_attr(path=f'Image_data')
        nol = attrs.get('Number_of_lines')
        nop = attrs.get('Number_of_pixels')
        img_slice = (slice(0, nol), slice(0, nop))

        if 'lon' in key.lower():
            stride = False
            if np.abs(np.nanmin(data) - np.nanmax(data)) > 180.:
                stride = True
                data[data < 0] = 360. + data[data < 0]
            data = interp2d(array=data, interval=interval)[img_slice]
            if stride:
                data[data > 180.] = data[data > 180.] - 360.
            return data
        # Get Latitude
        return interp2d(array=data, interval=interval)[img_slice]

    def get_proj(self, lon_0: float = None, lat_0: float = None):
        if lon_0 is None:
            if self.lon is None:
                key = 'Longitude' \
                    if self.file.suffix == '.h5' \
                    else 'longitude'
                lon_0 = self.get_geo(key=key).mean()
            else:
                lon_0 = self.lon.mean()

            if self.lat is None:
                key = 'Latitude' \
                    if self.file.suffix == '.h5' \
                    else 'latitude'
                lat_0 = self.get_geo(key=key).mean()
            else:
                lat_0 = self.lat.mean()
        projection = dict(datum=self.datum,
                          lat_0=lat_0,
                          lon_0=lon_0,
                          proj=self.proj_name,
                          units='m')
        return pyproj.Proj(projection)

    def get_area_def(self, area_extent: tuple, shape: tuple):
        height, width = shape
        return AreaDefinition(area_id=self.area_id,
                              description=self.description,
                              proj_id=self.proj.name,
                              projection=self.proj.srs,
                              width=width,
                              height=height,
                              area_extent=area_extent)

    def get_extent(self):
        return self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()

    def get_flag_meanings(self, key='QA_flag'):
        attrs = self.fmt_attr(path=f'Image_data/{key}')

        def flag_meaning(flg: str, j: int):
            start = len(f'Bit-{j}) ')
            end = flg.index(': ')
            return flg[start:end]

        return [flag_meaning(flg=flag, j=i)
                for i, flag in enumerate(attrs.pop('Data_description').split('\n'))
                if len(flag) > 0]

    # ----------
    # Attributes
    # ----------

    def nc_attrs(self, key: str = None):
        attrs = self.glob_attrs.copy()
        if key:
            key_attrs = {at: self.path[key].getncattr(at)
                         for at in self.path[key].ncattrs()}
            units = [key for key in key_attrs.keys()
                     if 'unit' in key.lower()]
            if len(units) > 0:
                if units[0] == 'Unit':
                    unit = key_attrs.pop(units[0])
                    key_attrs['units'] = unit
            if len(units) == 0:
                key_attrs['units'] = 'NA'
            attrs.update({key: key_attrs})
        return attrs

    def h5_attrs(self, key: str = None):
        attrs = self.glob_attrs.copy()
        if key is None:
            return attrs
        path = f'Image_data/{key}'
        if key[:3].lower() in ('lon', 'lat'):
            path = f'Geometry_data/{key}'
        key_attrs = self.fmt_attr(path=path)
        if 'Unit' in key_attrs.keys():
            unit = key_attrs.pop('Unit')
            key_attrs['units'] = unit
        attrs.update({key: key_attrs})
        return attrs

    def fmt_attr(self, path: str):
        result = {}
        for key, val in self.obj[path].attrs.items():
            if key in ('Dim0', 'Dim1'):
                continue
            try:
                val = val[0]
            except IndexError:
                pass
            if type(val) in (bytes, np.bytes_):
                val = val.decode()
            result.update({key: val})
        desc = result['Data_description'] \
            if 'Data_description' in result.keys() else None
        if desc and ('Remote Sensing Reflectance(Rrs)' in desc):
            result['units'] = result['Rrs_unit']
        return result

    def spatial_resolution(self):
        attrs = self.h5_attrs() \
            if self.file.suffix == '.h5' \
            else self.nc_attrs()
        spr = None

        if 'spatialResolution' in attrs.keys():
            spr = attrs.pop('spatialResolution')
        if 'spatial_resolution' in attrs.keys():
            spr = attrs.pop('spatial_resolution')
        if self.file.suffix == '.h5' and \
                ('Q_' in self.file.name):
            spr = '250 m'
        if self.file.suffix == '.h5' and \
                ('K_' in self.file.name):
            spr = '1 km'

        unit = ''.join(re.findall('[a-z]', spr, re.IGNORECASE))
        spr = float(spr.strip(unit))
        if unit.lower() == 'km':
            spr *= 1000
        return spr

    # ----------
    # Close File
    # ----------
    def close(self, obj=None):
        if obj is None:
            self.obj.close()
            [setattr(self, key, None)
             for key in self.__dict__.keys()]
            return
        obj.close()
        return


def get_norm(vmin: float, vmax: float, scale: str):
    if scale == 'log':
        return LogNorm(vmin=vmin, vmax=vmax)
    levels = MaxNLocator(nbins=256).tick_values(vmin=vmin, vmax=vmax)
    return BoundaryNorm(levels, ncolors=256, clip=True)


def elapsed(start: float, file: Path):
    # ------
    # Report
    # ------
    t = time.process_time() - start
    info = colored(file.name, color='green')
    if t > 60:
        mn = int(t // 60)
        sc = int(t % 60)
        print(f'{info} | {mn:2} min {sc:02} sec')
    else:
        print(f'{info} | {t:.3f} sec')
    return


def resample(sds, extent, obj):
    """
    :param sds:
    :type sds:
    :param extent:
    :type extent:
    :param obj:
    :type obj:
    :return:
    :rtype:
    """
    roi = obj.spatial_resolution()

    # ------------
    # geo-loc bbox
    # ------------
    lon_box = extent[0], extent[2]
    lat_box = extent[1], extent[3]
    lat_ts = np.mean(lat_box)

    datum = 'WGS84'
    # Get the step in degrees from the resolution in meters
    g = pyproj.Geod(ellps=datum)
    # define the lon/lat degree step from resolution
    dy = round((roi * 360.) / (2. * np.pi * g.b) * 1e7) / 1e7
    dx = round((roi * 360.) / (2. * np.pi * g.a * np.cos(np.deg2rad(lat_ts))) * 1e7) / 1e7

    # Grid of lon/lat
    w = np.arange(extent[0], extent[2], dx, np.float32).size
    h = np.arange(extent[3], extent[1], -dy, np.float32).size

    # -----------
    # pyproj proj
    # -----------
    obj.proj = obj.get_proj(lon_0=np.mean(lon_box), lat_0=lat_ts)
    target_geo = obj.get_area_def(area_extent=extent, shape=(h, w))

    # --------
    # area def
    # --------
    source_geo = SwathDefinition(obj.lon, obj.lat)
    result = resample_nearest(source_geo_def=source_geo,
                              data=sds,
                              target_geo_def=target_geo,
                              radius_of_influence=roi * 2,
                              fill_value=None)
    return result, target_geo


def quicklook(file, key, outpath, scale, bbox=None):
    """
    Saves a quick view of satellite swath image in a figure.
    The swath data is resampled into the bbox if provided.

    :param file: filename from where to obtain the swath data
    :type file: Path
    :param key: variable being read
    :type key: str
    :param outpath: output filename
    :type outpath: Path
    :param scale: lin or log. Indicates the colour scale to use. Chlorophyll has log-normal scale.
    :type scale: str
    :param bbox: bounding box of the area of interest as [lon_min, lat_min, lon_max, lat_max].
    :type bbox: tuple
    :return: None
    :rtype: None
    """

    proj_name = 'lonlat' if bbox else 'laea'
    kw = {'var': key, 'proj_name': proj_name}
    with File(file=file, mode='r', **kw) as obj:
        # ---------
        # read data
        # ---------
        sds = obj.get_data()
        crs = ccrs.PlateCarree()
        lon, lat = obj.lon, obj.lat
        if bbox is not None:
            sds, tg = resample(sds=sds, extent=bbox, obj=obj)
            crs = tg.to_cartopy_crs()
            lon, lat = tg.get_lonlats()

    # ---------
    # Get image
    # ---------
    size = [s / 100 for s in sds.shape[::-1]]
    fig, ax = plt.subplots(figsize=size,
                           clear=True,
                           num=1,
                           subplot_kw={'projection': crs})
    # --------
    # img norm
    # --------
    mn, mx = np.ma.min(sds), np.ma.max(sds)
    norm = get_norm(vmin=mn, vmax=mx, scale=scale)
    cmp = plt.get_cmap('nipy_spectral')

    # --------
    # img disp
    # --------
    ax.pcolormesh(lon,
                  lat,
                  sds,
                  shading='auto',
                  norm=norm,
                  cmap=cmp)
    ax.gridlines(color='gray', linestyle=':')
    ax.axis('off')
    ax.coastlines(color='w')

    fig.set_facecolor("black")
    fig.savefig(outpath)
    plt.close(fig)
    return


def integral(val):
    val = Decimal(val)
    if val == val.to_integral():
        return f'{val.quantize(Decimal(1))}'
    return f'{val.normalize()}'


def get(file_pattern, key, bbox=None):
    """
    Get a quick view of satellite swath image
    :param file_pattern: File pattern or name
    :type file_pattern: Path
    :param key: variable to be displayed
    :type key: str
    :param bbox: bounding box of the area of interest as [lon_min, lat_min, lon_max, lat_max].
    :type bbox: tuple
    :return: None
    :rtype: None
    """
    plt.rcParams.update({
        'figure.frameon': False,
        'figure.figsize': (100, 100),
        'figure.constrained_layout.use': True
    })

    join = '' if bbox is None else '_'.join(integral(val=f'{v}') for v in bbox)

    for f in file_pattern.parent.glob(file_pattern.name):
        start = time.process_time()

        ext = f.suffix
        out = f.parent.joinpath(
            f.name.replace(ext, f'{join}_{key}.png'))
        if out.is_file():
            elapsed(start=start, file=out)
            continue
        # ------
        # ql fig
        # ------
        scale = 'log' if key in ('CHLA', 'chlor_a', 'CDOM') else 'lin'
        quicklook(file=f, outpath=out, scale=scale, key=key, bbox=bbox)
        # -------
        elapsed(start=start, file=out)
    return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description='''\
      Save browse images from Level-2 *.nc or *.h5 files in PNG format.

      Outputs:
         PNG image files from input pattern. 
            Default is to save of chlor_a (.nc) or CHLA (.h5).
            Otherwise use variable if --variable is specify. 

      Inputs:
        The argument-list is a set of --keyword value pairs (see optional arguments below).

        * Compatibility: This script was developed with Python 3.9.

      ''', epilog=textwrap.dedent('''\
        Type python savefig.py --help for details.
        --------
        Examples usage calls:
            python savefig.py GC1SG1_2022*_IWPRQ_3000.h5 
            python savefig.py GC1SG1_2022*_IWPRQ_3000.h5 --variable=CDOM
            python savefig.py A2022*.L2_LAC_OC.x.nc
            python savefig.py A2022*.L2_LAC_OC.x.nc --variable=Rrs_412
            
                        '''), add_help=True)

    parser.add_argument('--pattern', nargs=1, type=str, required=True, help='''\
      Valid file pattern of level-2 file
      File must be one of the two supported formats (NASA netcdf file or JAXA hdf5 file.
      ''')

    parser.add_argument('--variable', nargs=1, default=[None], type=str, help=('''\
      Satellite variable name to be used and contained in the input file
      OPTIONAL: default use chlor_a or CHLA depending on file type
      Use with --pattern
      '''))

    parser.add_argument('--bbox', nargs=4, default=[None],
                        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'),
                        type=float, help=('''\
      Bounding box of the area of interest. 
      OPTIONAL: default use input image bbox
      '''))

    args = vars(parser.parse_args())
    pat = Path(args.get('pattern')[0])

    var = args.get('variable')[0]
    if ('h5' in pat.name) and (var is None):
        var = 'CHLA'
    if ('nc' in pat.name) and (var is None):
        var = 'chlor_a'

    get(key=var, file_pattern=pat, bbox=args.get('bbox'))
