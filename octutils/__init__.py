import subprocess
import time
from datetime import datetime
from netrc import netrc, NetrcParseError
from pathlib import Path

import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import numpy as np
import requests
from cartopy import crs as ccrs
from shapely.geometry import Polygon

from . import quicklook


def getauth(host: str, raise_errors=None):
    """
    Fetches the username and password from the netrc file

    Parameters
    ----------

    @param host: url of the host for which user and pass are needed
    @param raise_errors: [optional] error handler

    Returns
    ----------
    @return user, passwd
    """

    file = Path().home().joinpath('.netrc')

    if not file.is_file():
        print('.netrc fileNotFound\nMake sure file exists')
        return [None] * 2

    try:
        _netrc = netrc(file).authenticators(host)
        if _netrc:
            # Return with login / password
            return _netrc[0], _netrc[-1]

    except (NetrcParseError, IOError):
        if raise_errors:
            raise


def retrieve(url: str, session: requests):
    get_file = Path(url).name

    response = session.get(url,
                           allow_redirects=True,
                           stream=True,
                           timeout=(10, 10))
    print(f'RequestsURL: {response.url}')
    total = int(response.headers.get('content-length'))

    if response.ok:
        from tqdm import tqdm
        print('\nDownloading {}'.format(get_file))
        with tqdm(total=total) as bar, open(get_file, "wb") as handle:
            for chunk in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                # download progress check tqdm
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
                    time.sleep(0.1)
                    bar.update(len(chunk))
        return 0


def attr_fmt(h5: h5py, address: str):
    result = {}
    for key, val in h5[address].attrs.items():
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


def search(bbox: tuple, start: datetime, end: datetime, dtype: str) -> list:
    """
    Search files within the bounding box (bbox) and return the list of files in the server

    :param bbox:
    :type bbox:
    :param start:
    :type start:
    :param end:
    :type end:
    :param dtype:
    :type dtype:
    :return: a list of lists with [[fullpath, filename], ...] of product files
    :rtype:
    """
    did = '10002000,10002001,10002002'
    product = ''
    if dtype.lower() == 'rrs':
        product = 'NWLR'
        did = '10002000'
    if dtype.lower() == 'oc':
        product = 'IWPR'
        did = '10002001'
    if dtype.lower() == 'sst':
        did = '10002002'
        product = 'SST'

    w, s, e, n = bbox
    bounds = Polygon(((w, s), (w, n), (e, n), (e, s), (w, s)))
    bbox = f"&bbox={w},{s},{e},{n}"
    pcl, _, _ = '&pslv=L2', '&resolution=250.0', '&sat=GCOM-C'
    gdi, count, _ = f'&datasetId={did}', '&count=1000', '&sen=SGLI'
    std = f'&startTime={start.strftime("%Y-%m-%d")}T00:00:00Z'
    end = f'&endTime={end.strftime("%Y-%m-%d")}T23:59:59Z'
    browser = 'https://gportal.jaxa.jp/csw/csw?service=CSW&version=3.0.0&request=GetRecords'

    # SGLI data query.
    query = f'{browser}&outputFormat=application/json' \
            f'{gdi}{bbox}{std}{end}{count}{pcl}'

    response = requests.get(query)
    filename_list = []
    append = filename_list.append

    for attrs in response.json()['features']:
        size = attrs['properties']['product']['size']
        file = attrs['properties']['product']['fileName']
        ply = Polygon(np.squeeze(np.array(attrs['geometry']['coordinates'])))
        itsc = ply.intersects(bounds)

        fi = Path(file).name
        lru = file[:-len(fi) - 1]

        if ('OCEAN' in file) and \
                ('Q_3' in fi) and \
                (int(size) > 0) \
                and (itsc is True):

            if product in fi:
                append(f"{lru[lru.find('standard'):]}/{fi}")
    return filename_list


def getfile(bbox, start_date, end_date, output_dir=None, sensor='sgli', dtype='OC'):
    """

    :param bbox: bounding box (area of interest) e.g., (lon_min, lat_min, lon_max, lat_max)
                 Mozambique channel bbox (30, -30, 50, -10)
    :type bbox: tuple
    :param start_date: data search start date
    :type start_date: datetime
    :param end_date: data search end date
    :type end_date: datetime
    :param sensor: sensor name, currently only SGLI is support.
                   For NASA's supported sensor is easy to use the OceanColor Web to place data orders.
    :param output_dir: if output path is not specified, the current working directory is used.
                       A folder with the sensor name and `L2` appended is created and data is saved therein
    :type output_dir: Path
    :type sensor: str
    :param dtype: data type, OC - ocean colour or SST - sea surface temperature
    :type dtype: str
    :return: list of images with download links
    :rtype: list
    """
    if sensor != 'sgli':
        return []

    files = search(bbox=bbox,
                   start=start_date,
                   end=end_date,
                   dtype=dtype)

    user, passwd = getauth(host='ftp.gportal.jaxa.jp')
    downloaded = []
    fetched = downloaded.append
    if output_dir is None:
        output_dir = Path.cwd().joinpath(f'{sensor}_L2'.upper())
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    for src_file in files:

        cmd = f'wget -nc --preserve-permissions ' \
              f'--remove-listing --tries=5 ' \
              f'ftp://{user}:{passwd}@' \
              f'ftp.gportal.jaxa.jp/{src_file} ' \
              f'--directory-prefix={output_dir}'

        name = output_dir.joinpath(Path(src_file).name)
        status = subprocess.call(cmd, shell=True)
        if (status == 0) and name.is_file():
            fetched(name)
    return downloaded


def get_data(file: h5py.File, key: str):
    """
    Converts digital number to geophysical values in an open hdf5 `h5`
    :param file: open h5py.File object
    :type file: h5py
    :param key: name of the variable being read
    :type key: str
    :return: geophysical data
    :rtype: np.array
    """
    if key == 'QA_flag':
        sds = np.ma.squeeze(h5[f'Image_data/{key}'][:])
        np.ma.set_fill_value(sds, 0)
        return sds

    path = f'Geometry_data/{key}' \
        if key[:3].lower() in ('lon', 'lat') \
        else f'Image_data/{key}'
    fill_value = np.float32(-32767)
    sdn = file[path][:]
    attrs = attr_fmt(h5=file, address=path)

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
        # https://shikisai.jaxa.jp/faq/docs/GCOM-C_Products_Users_Guide_entrylevel__attach4_jp_191007.pdf#page=46
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
    sds = sdn * slope + offset

    if key[:3].lower() in ('lon', 'lat'):
        return sds
    return np.ma.masked_array(sds,
                              mask=mask,
                              fill_value=fill_value,
                              dtype=np.float32)


def interp2d(src_geo: np.array, interval: int):
    """
        Bilinear interpolation of SGLI geo-location corners to a spatial grid
        (c) Adapted from the GCOM-C PI kick-off meeting 201908, K. Ogata (JAXA)

        Parameters
        ----------
        src_geo: np.array
            either lon or lat
        interval: int
            resampling interval in pixels

        Return
        ------
        out_geo: np.array
            2-D array with dims == to geophysical variables
    """
    sds = np.concatenate((src_geo, src_geo[-1].reshape(1, -1)), axis=0)
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


def geometry_data(file: h5py.File, key: str):
    """
    Retrieve SGLI navigation data
    (c) GCOM-C PI kick-off meeting 201908, K. Ogata (JAXA)
    :param file: open h5py.File object
    :type file: h5py.File
    :param key: variable name
    :type key: str
    :return: geolocation data
    :rtype: np.array
    """
    nsl, = file['Image_data'].attrs['Number_of_lines']
    psl, = file['Image_data'].attrs['Number_of_pixels']
    img_size = (slice(0, nsl), slice(0, psl))

    data = get_data(file=file, key='Latitude')
    interval, = file[f'Geometry_data/{key}'].attrs['Resampling_interval']
    if 'lat' in key.lower():
        # Get Latitude
        return interp2d(src_geo=data, interval=interval)[img_size]

    # Get Longitude
    is_stride_180 = False
    if np.abs(np.nanmin(data) - np.nanmax(data)) > 180.:
        is_stride_180 = True
        data[data < 0] = 360. + data[data < 0]
    data = interp2d(src_geo=data, interval=interval)[img_size]

    if is_stride_180:
        data[data > 180.] = data[data > 180.] - 360.
    return data


def imview(data, lon, lat, scale='lin'):
    """
    Quick view of satellite swath image
    :param data:
    :type data:
    :param lon:
    :type lon:
    :param lat:
    :type lat:
    :param scale:
    :type scale:
    :return:
    :rtype:
    """

    y, x = data.shape
    x, y = 20, 20 * (y / x)
    crs = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(x, y),
                           subplot_kw={'projection': crs})
    # --------
    # img norm
    # --------
    mn, mx = np.ma.min(data), np.ma.max(data)
    norm = quicklook.get_norm(vmin=mn, vmax=mx,
                              scale=scale, caller='imv')
    cmp = plt.get_cmap('nipy_spectral')

    # --------
    # img disp
    # --------
    extent = lon.min(), lon.max(), lat.min(), lat.max()
    m = ax.imshow(data,
                  transform=crs,
                  extent=extent,
                  norm=norm,
                  cmap=cmp,
                  origin='upper')
    # ax.axis('off')
    ax.gridlines()
    ax.set_aspect(1 / np.cos(np.deg2rad(lat.mean())))
    fig.colorbar(m, ax=ax, shrink=0.5, pad=0.01)

    land = cfeature.NaturalEarthFeature(
        'physical', 'land', '10m',
        linewidth=2,
        edgecolor='r',
        facecolor='0.3')
    ax.add_feature(land)
    return fig, ax


__all__ = ['getfile', 'quicklook', 'get_data', 'geometry_data', 'imview']
