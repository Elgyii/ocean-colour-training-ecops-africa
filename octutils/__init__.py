__author__ = "Eligio Maure"
__copyright__ = "Copyright (C) 2022 Eligio Maure"
__license__ = "MIT"

import os
import shutil
import subprocess
import time
from contextlib import closing
from datetime import datetime
from netrc import netrc, NetrcParseError
from pathlib import Path
from urllib import request

import h5py
import numpy as np
import requests
from matplotlib import colors, ticker
import termcolor
from shapely.geometry import Polygon

from . import level2
from .pyutils import (
    pyextract,
    add_marker,
    get_composite,
    annual_max,
    annual_min
)
from .level2 import File
from .level2 import l2remap

SATELLITES = {
    'czcs': {'INSTRUMENT': 'CZCS',
             'PLATFORM': 'Nimbus-7',
             'SEARCH': 'CZCS_L2_',
             'SENSOR': '',
             'PERIOD_END': datetime(1978, 10, 30).toordinal(),
             'PERIOD_START': datetime(1986, 6, 22).toordinal()},

    'goci': {'INSTRUMENT': 'GOCI',
             'PLATFORM': 'COMS',
             'SEARCH': 'GOCI_L2_',
             'SENSOR': '',
             'PERIOD_END': datetime(2021, 4, 1).toordinal(),
             'PERIOD_START': datetime(2011, 4, 1).toordinal()},

    'meris': {'INSTRUMENT': 'MERIS',
              'PLATFORM': 'ENVISAT',
              'SEARCH': 'MERIS_L2_',
              'SENSOR': 'merr',
              'PERIOD_END': datetime(2012, 4, 8).toordinal(),
              'PERIOD_START': datetime(2002, 4, 29).toordinal()},

    'modisa': {'INSTRUMENT': 'MODIS',
               'PLATFORM': 'AQUA',
               'SEARCH': 'MODISA_L2_',
               'SENSOR': 'amod',
               'PERIOD_END': datetime.today().toordinal(),
               'PERIOD_START': datetime(2002, 7, 4).toordinal()},

    'modist': {'INSTRUMENT': 'MODIS',
               'PLATFORM': 'TERRA',
               'SEARCH': 'MODIST_L2_',
               'SENSOR': 'tmod',
               'PERIOD_END': datetime.today().toordinal(),
               'PERIOD_START': datetime(2000, 2, 24).toordinal()},

    'octs': {'INSTRUMENT': 'OCTS',
             'PLATFORM': 'ADEOS-I',
             'SEARCH': 'OCTS_L2_',
             'SENSOR': '',
             'PERIOD_END': datetime(1997, 6, 29).toordinal(),
             'PERIOD_START': datetime(1996, 10, 31).toordinal()},

    'seawifs': {'INSTRUMENT': 'SeaWiFS',
                'PLATFORM': 'OrbView-2',
                'SEARCH': 'SeaWiFS_L2_',
                'SENSOR': '',
                'PERIOD_END': datetime(2010, 12, 11).toordinal(),
                'PERIOD_START': datetime(1997, 9, 4).toordinal()},

    'sgli': {'INSTRUMENT': 'SGLI',
             'PLATFORM': 'GCOM-C',
             'SEARCH': 'GC1SG1_*_L2SG_*_',
             'SENSOR': '',
             'PERIOD_END': datetime.today().toordinal(),
             'PERIOD_START': datetime(2018, 1, 1).toordinal()},

    'viirsn': {'INSTRUMENT': 'VIIRS',
               'PLATFORM': 'NPP',
               'SEARCH': 'VIIRSN_L2_',
               'SENSOR': 'vrsn',
               'PERIOD_END': datetime.today().toordinal(),
               'PERIOD_START': datetime(2012, 1, 2).toordinal()}
}


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


def retrieve(url, dst, user, passwd):
    file = Path(url).name

    session = requests.Session()
    session.auth = (user, passwd)
    response = session.get(url,
                           allow_redirects=True,
                           stream=True,
                           timeout=(10, 10))
    print(f'RequestsURL: {response.url}')
    total = int(response.headers.get('content-length'))

    if response.ok:
        from tqdm import tqdm
        print('\nDownloading {}'.format(file))
        with tqdm(total=total) as bar, open(dst, "wb") as handle:
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


def wget_file(odir, manifest=None, url=None):
    home = Path().home().absolute()
    cookies = home.joinpath('.urs_cookies')
    user, passwd = getauth(host='urs.earthdata.nasa.gov')

    cmd = 'wget -nc ' \
          '--auth-no-challenge=on ' \
          '--keep-session-cookies ' \
          f'--load-cookies {cookies} ' \
          f'--save-cookies {cookies} ' \
          f'--directory-prefix={odir}'

    if manifest is None:
        cmd = f'{cmd} --content-disposition "{url}"'
    else:
        cmd = f'{cmd} --content-disposition -i {manifest}'
    if os.name == 'nt':
        cmd = f'{cmd} --user={user} --password={passwd}'
    return subprocess.call(cmd, shell=True)


def csw_search(bbox, start, end, dtype):
    """
    Search files within the bounding box (bbox) and return the list of files in the server

    :param bbox: bounding box (area of interest) e.g., (lon_min, lat_min, lon_max, lat_max)
    :type bbox: tuple
    :param start: data search start date
    :type start: datetime
    :param end: data search end date
    :type end: datetime
    :param dtype: data type, OC - ocean colour, SST - sea surface temperature, rrs for remote sensing reflectance
    :type dtype: str
    :return: a list of lists of [url0, url1, ...] product files to download
    :rtype: list
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

    download_list = []
    append = download_list.append

    response = requests.get(query)
    features = response.json()['features']

    for attrs in features:
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
    return download_list


def cmr_search(sensor, bbox, start, end, dtype, odir):
    """
    Search files within the bounding box (bbox) and return the list of files in the server

    :param bbox: bounding box (area of interest) e.g., (lon_min, lat_min, lon_max, lat_max)
    :type bbox: tuple
    :param odir: if output path is not specified, the current working directory is used.
    :type odir: Path
    :param sensor: sensor name, 'czcs', 'goci', 'meris', 'modisa', 'modist', 'octs', 'seawifs', 'sgli', 'viirsn'
    :type sensor: str
    :param start: data search start date
    :type start: datetime
    :param end: data search end date
    :type end: datetime
    :param dtype: data type, OC - ocean colour, SST - sea surface temperature
    :type dtype: str
    :return: a list of lists with [url0, url1, ...]
    :rtype: list
    """

    user, passwd = getauth(host='urs.earthdata.nasa.gov')
    bounding_box = ','.join(f'{v}' for v in bbox)
    instrument = SATELLITES[sensor]['INSTRUMENT']
    platform = SATELLITES[sensor]['PLATFORM']

    short_name = f"{SATELLITES[sensor]['SEARCH']}{dtype.upper()}"
    if dtype.lower() == 'sst':
        platform, instrument = (f'S{platform}', instrument) \
            if sensor == 'viirsn' else (platform, instrument)
        short_name = f'{platform}_{instrument}*.L2.SST.nc'

    query = "https://cmr.earthdata.nasa.gov/search/granules.json?page_size=2000" \
            "&provider=OB_DAAC" \
            f"&bounding_box={bounding_box}" \
            f"&instrument={instrument}" \
            f"&platform={platform}" \
            f"&short_name={short_name}" \
            f"&temporal={start.strftime('%Y-%m-%dT%H:%M:%SZ')}," \
            f"{end.strftime('%Y-%m-%dT%H:%M:%SZ')}&" \
            "sort_key=short_name" \
            "&options[short_name][pattern]=true"

    response = requests.get(query)
    if response.status_code != 200:
        print(f'requests status {response.status_code}')
        return []

    content = response.json()
    download_files = []
    append = download_files.append

    for entry in content['feed']['entry']:
        granid = entry['producer_granule_id']
        file = odir.joinpath(granid)
        geturl = entry['links'][0]['href']
        try:
            status = wget_file(url=geturl, odir=odir)
        except Exception as e:
            print(f'WGET Failed, using requests\n{e}')
            status = retrieve(url=geturl, dst=file, user=user, passwd=passwd)
        if status == 0:
            append(file)
    return download_files


def getfile(sensor, dtype='OC', start_date=None, end_date=None,
            bbox=None, output_dir=None, manifest=None):
    """
    Retrieves data from NASA's OceanColourWeb and JAXA's G-Portal.
    For NASA it can use http_manifest.txt or ordinary search with bbox and time interval and sensor name.
    G-Portal is searched and downloaded using bbox, sgli and temporal interval.

    Parameters
    ----------
    sensor: str
        sensor name, 'sgli', 'czcs', 'goci', 'meris', 'modisa', 'modist', 'octs', 'seawifs', 'sgli', 'viirsn'
    dtype: str
        data type, OC - ocean colour, SST - sea surface temperature, rrs for remote sensing reflectance
    start_date: datetime
        data search start date
    end_date: datetime
        data search end date
    bbox: tuple
        bounding box (area of interest) e.g., (lon_min, lat_min, lon_max, lat_max)
        Mozambique channel bbox (30, -30, 50, -10)
    output_dir: Path
        if output path is not specified, the current working directory is used.
        A folder with the sensor name and `L2` appended is created and data is saved therein
    manifest: Path
        a manifest file with download urls

    Returns
    -------
    files: list
        list of downloaded data files
    """
    downloaded = []
    fetched = downloaded.append
    if output_dir is None:
        output_dir = Path.cwd().joinpath('data')
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    # check is urs_cookies exists
    home = Path().home().absolute()
    cookies = home.joinpath('.urs_cookies')
    netrc_file = home.joinpath('.netrc')

    if not cookies.is_file():
        with open(cookies, 'w') as txt:
            txt.writelines('')

    if not netrc_file.is_file():
        with open(netrc_file, 'w') as txt:
            name = termcolor.colored('USERNAME', color='magenta', attrs=['bold'])
            pw = termcolor.colored('PASSWORD', color='red', attrs=['bold'])
            center = termcolor.colored('NASA EARTHDATA', color='blue', attrs=['bold'])
            user = input(f'Type in your {center} {name} and hit ENTER: ')
            passwd = input(f'Type in your {center} {pw} and hit ENTER: ')
            txt.writelines(f'machine urs.earthdata.nasa.gov login {user} password {passwd}\n')

            center = termcolor.colored('JAXA G-PORTAL', color='green', attrs=['bold'])
            user = input(f'Type in your {center} {name} and hit ENTER: ')
            txt.writelines(f'machine ftp.gportal.jaxa.jp login {user} password anonymous\n')
            print(f'Data save in {netrc_file}')
        if os.name != 'nt':
            subprocess.call(f'chmod  0600 {netrc_file}', shell=True)

    if sensor != 'sgli':
        if manifest is None:
            files = cmr_search(bbox=bbox, start=start_date, end=end_date,
                               sensor=sensor, odir=output_dir, dtype=dtype)
        else:
            wget_file(manifest=manifest, odir=output_dir)
            return list(output_dir.glob('requested_file*'))
        return files

    files = csw_search(bbox=bbox, start=start_date, end=end_date, dtype=dtype)
    user, passwd = getauth(host='ftp.gportal.jaxa.jp')

    for src_file in files:

        url = f'ftp://{user}:{passwd}@ftp.gportal.jaxa.jp/{src_file}'
        cmd = f'wget -nc --preserve-permissions --remove-listing --tries=5 ' \
              f'{url} --directory-prefix={output_dir}'

        name = output_dir.joinpath(Path(src_file).name)
        try:
            status = subprocess.call(cmd, shell=True)
        except Exception as e:
            print(f'WGET Failed, using urllib\n{e}')
            with closing(request.urlopen(url)) as r:
                with open(name, 'wb') as f:
                    shutil.copyfileobj(r, f)
            status = 0
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


def progress_bar(name, count, total):
    block = int(20 * count // total)
    bar = '█' * block + '-' * (20 - block)
    percent = f'{(100 * count / total):.1f}'
    status = f'{name} | {percent}%'
    print(f'\r{status} |{bar}| {count}/{total}', end='')


__all__ = [
    'File',
    'add_marker',
    'annual_max',
    'annual_min',
    'getfile',
    'get_composite',
    'pyextract',
    'level2',
    'l2remap',
    'progress_bar'
]
