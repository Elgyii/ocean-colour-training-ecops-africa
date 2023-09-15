import re
import traceback
from pathlib import Path

import h5py
import numpy as np
import pyproj
from dateutil.parser import parse
from netCDF4 import Dataset
from pyresample import create_area_def
from pyresample.geometry import SwathDefinition, AreaDefinition
from pyresample.kd_tree import resample_nearest


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
        Interface for data I/O (JAXA's hdf5 or NASA's netcdf4)
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
        self.area_id = 'ease_sh'  # The Equal-Area Scalable Earth (EASE), https://nsidc.org/data/ease
        self.description = 'Antarctic EASE grid'
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
            raise KeyError('Variable should not be "None"')
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
        """

        """
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
        """
        Returns a uniformly spaced geographic region to be used in resampling (the target area).
        The projection for an area is typically described by longitude/latitude coordinates or
        described in X/Y coordinates in meters.
        see https://pyresample.readthedocs.io/en/latest/geo_def.html#areadefinition

        Parameters
        ----------
        shape: tuple
            width: int
               Number of grid columns
            height: int
               Number of grid rows
        area_extent: tuple
           lower_left_x, lower_left_y, upper_right_x, upper_right_y
           where
               lower_left_x: projection x coordinate of lower left corner of lower left pixel
               lower_left_y: projection y coordinate of lower left corner of lower left pixel
               upper_right_x: projection x coordinate of upper right corner of upper right pixel
               upper_right_y: projection y coordinate of upper right corner of upper right pixel

        Returns
        -------
        areaDefinition: AreaDefinition
           Pyresample AreaDefinition
        """
        height, width = shape
        return AreaDefinition(area_id=self.area_id,
                              description=self.description,
                              proj_id=self.proj.name,
                              projection=self.proj.srs,
                              width=width,
                              height=height,
                              area_extent=area_extent)

    def get_area_definition(self, area_extent,
                            resolution, units='metres',
                            width=None, height=None):
        """
        Returns a uniformly spaced geographic region to be used in resampling (the target area).
        The projection for an area is typically described by longitude/latitude coordinates or
        described in X/Y coordinates in meters.
        see https://pyresample.readthedocs.io/en/latest/geo_def.html#areadefinition

        Parameters
        ----------
        area_extent: tuple
            lower_left_x, lower_left_y, upper_right_x, upper_right_y
            where
            lower_left_x: projection x coordinate of lower left corner of lower left pixel
            lower_left_y: projection y coordinate of lower left corner of lower left pixel
            upper_right_x: projection x coordinate of upper right corner of upper right pixel
            upper_right_y: projection y coordinate of upper right corner of upper right pixel
        units: str
            projection units. Can be one of 'deg', 'degrees', 'meters', 'metres'
        resolution: float

        width: str
            number of pixels in the x direction
        height: str
            number of pixels in the y direction

        Returns
        -------
        area-definition: AreaDefinition
            Pyresample AreaDefinition
        """
        return create_area_def(area_id=self.area_id,
                               projection=self.proj.srs,
                               area_extent=area_extent,
                               resolution=resolution,
                               units=units,
                               height=height,
                               width=width)

    def get_extent(self):
        return self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()

    def get_flag_meanings(self, key='QA_flag'):
        if self.file.suffix == '.nc':
            return self.path['l2_flags'].flag_meanings.split()

        attrs = self.fmt_attr(path=f'Image_data/{key}')

        if 'SST' in self.file.name:
            def flag_meaning(flg: str, j: int):
                try:
                    start = len(f'Bit-{j}:') if 'Bit' in flg else flg.index(f'{j}:')
                    if start == 0:
                        start = len(f'{j}:')
                except ValueError:
                    if '6-7' in flg:
                        return [flg[len('6-7:'):].upper(), flg[len('6-7:'):].upper()]
                    if '8:' in flg:
                        return [flg[len('8:'):].upper()]
                    if ('1:' in flg) and (j == 8):
                        return [''.join([return_flags.pop(-1), ' ', flg.upper()])]
                    start = len(f'{j - 1}:')
                    print(j)
                return [flg[start:].upper()]

            split_flag = attrs.pop('Data_description').split(',')

            return_flags = []
            extend = return_flags.extend
            for i, flag in enumerate(split_flag):
                if len(flag) > 0:
                    extend(flag_meaning(flg=flag, j=i))

            return return_flags

        def flag_meaning(flg: str, j: int):
            start = len(f'Bit-{j}) ')
            end = flg.index(': ')
            return flg[start:end]

        return [flag_meaning(flg=flag, j=i)
                for i, flag in enumerate(attrs.pop('Data_description').split('\n'))
                if len(flag) > 0]

    def screen_data(self, key, flags, show=True):
        if self.file.suffix == '.h5':
            l2flags = self.path['QA_flag'][:]
        else:
            l2flags = self.path['l2_flags'][:]
        data = self.get_data(key=key)
        mask, disp = 0, []
        fmean = self.get_flag_meanings()

        for j, bit in enumerate(flags):
            if bit == '1':
                mask += (int(bit) << j) & l2flags
            if bit == '1':
                disp.append(f'{fmean[j]}: {j}')
        if show:
            disp = '\n\t'.join(disp)
            info = f'{self.file.name} | {key}'
            print(f'{info}\n{"=" * len(info)}\nL2 FLAGS set to '
                  f'mask LAND and EXCLUDE low-quality pixels:\n\t{disp}')
        mask = np.where(mask > 0, 1, 0)
        return np.ma.masked_where(mask, data)
        # ----------

    # Attributes
    # ----------
    def get_attr(self, name: str, loc='/'):
        if self.file.suffix == '.nc':
            key = None if loc == '/' else loc
            attrs = self.nc_attrs(key=key)
            if 'time' in name:
                return parse(attrs.pop(name))
            return attrs.pop(name)

        if self.file.suffix == '.h5':
            key = None if loc == '/' else loc
            attrs = self.h5_attrs(key=key)
            name = 'Scene_start_time' \
                if name == 'time_coverage_start' \
                else name
            name = 'Scene_end_time' \
                if name == 'time_coverage_end' \
                else name
            if 'time' in name:
                return parse(attrs.pop(name))
            return attrs.pop(name)

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


def l2remap(file_pattern, key, flags, bbox, proj: str):
    """
    :param disp:
    :type disp:
    :param file_pattern:
    :type file_pattern: str
    :param flags:
    :type flags: str
    :param key:
    :type key: str
    :param bbox:
    :type bbox: tuple
    :param proj:
    :type proj: str
    :return:
    :rtype:
    """
    files = Path(file_pattern).parent.glob(
        Path(file_pattern).name)
    mapped = []
    append = mapped.append

    for i, file in enumerate(files):
        disp = True if i == 0 else False

        with File(file=file, mode='r') as fid:
            roi = fid.spatial_resolution()
            append(f'{file}')
            disp = True
            # ------------
            # geo-loc bbox
            # ------------
            lon_box = bbox[0], bbox[2]
            lat_box = bbox[1], bbox[3]
            lat_ts = np.mean(lat_box)
            lon_0 = float(np.mean(lon_box))

            # -----------
            # pyproj proj
            # -----------
            fid.proj = fid.get_proj(lon_0=lon_0, lat_0=float(lat_ts))
            (lower_left_x, lower_left_y) = fid.proj(bbox[0], bbox[1])
            (upper_right_x, upper_right_y) = fid.proj(bbox[2], bbox[3])
            area_extent = lower_left_x, lower_left_y, upper_right_x, upper_right_y
            target_geo = fid.get_area_definition(area_extent=area_extent, resolution=roi + 1)

            # ----------
            # screen sds
            # ----------
            sds = fid.screen_data(key=key, flags=flags, show=disp)

            # --------
            # area def
            # --------
            source_geo = SwathDefinition(fid.lon, fid.lat)
            result = resample_nearest(source_geo_def=source_geo,
                                      data=sds,
                                      target_geo_def=target_geo,
                                      radius_of_influence=roi * 2,
                                      fill_value=None)
            fid.data = result
    print('\n'.join(mapped))
    return
