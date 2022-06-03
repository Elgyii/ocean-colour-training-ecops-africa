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
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
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


class Status:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager
        https://stackoverflow.com/questions/22029562/python-how-to-make-simple-animated-loading-while-process-is-running

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        self.stop()


__all__ = ['getfile', 'quicklook']
