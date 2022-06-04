import subprocess
import time
from datetime import datetime
from netrc import netrc, NetrcParseError
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import requests
from cartopy import crs as ccrs
from shapely.geometry import Polygon
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
from . import quicklook
from .quicklook import File


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
    if 'sst' in dtype.lower():
        did = '10002002'
        product = dtype.upper()

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
    :return: list of downloaded data files
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
        output_dir = Path.cwd().joinpath('data')
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


def get_norm(vmin=0, vmax=1, nbins=256):
    levels = ticker.MaxNLocator(nbins=nbins).tick_values(vmin, vmax)
    return colors.BoundaryNorm(levels, ncolors=nbins, clip=True)


def custom_cmap(color_list: list = None):
    """

    :param color_list:
    :type color_list:
    :return:
    :rtype:
    """
    if color_list is None:
        color_list = ['r', 'g', 'b']

    return colors.LinearSegmentedColormap.from_list(
        'custom', color_list,
        # '#ff0033', '#ff8100', '#05832d', '#008bc1', '#4d0078'
        N=256)


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    https://github.com/scipy/scipy/blob/v0.18.1/scipy/misc/pilutil.py#L33-L100

    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0
    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


def enhance_image(image):
    """
    Color enhancement
    https://moonbooks.org/Codes/Plot-MODIS-granule-RGB-image-using-python-with-color-enhancement/
    http://www.idlcoyote.com/ip_tips/brightmodis.html
    """

    image = bytescale(image)

    x = np.array([0, 30, 60, 120, 190, 255], dtype=np.uint8)
    y = np.array([0, 90, 160, 210, 240, 255], dtype=np.uint8)

    along_track, cross_trak = image.shape

    scaled = np.zeros((along_track, cross_trak), dtype=np.uint8)
    for i in range(len(x) - 1):
        x1 = x[i]
        x2 = x[i + 1]
        y1 = y[i]
        y2 = y[i + 1]
        m = (y2 - y1) / float(x2 - x1)
        b = y2 - (m * x2)
        mask = ((image >= x1) & (image < x2))
        scaled = scaled + mask * np.asarray(m * image + b, dtype=np.uint8)

    mask = image >= y[-1]
    scaled = scaled + (mask * 255)
    return scaled


__all__ = ['getfile', 'quicklook', 'File', 'natural_color']
