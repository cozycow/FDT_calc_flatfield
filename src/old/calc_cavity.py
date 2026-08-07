from os import path
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
from scipy.ndimage import gaussian_filter, binary_dilation

from prefilter_correction import correct_prefilter
from limb_fitting import *
from utils import *


def calc_cavity(files, folder_out='',
                   dark_file=None,
                   deadpix_file=None,
                   prefilter_file=None,
                   distortion_file=None,
                   niter=10,
                   double_pass=True,
                   quicklook=True,
                   verbose=True):

    '''
    :param files: list of input files paths or path to input files folder
    :param folder_out: output folder path
    :param dark_file: path to dark signal file
    :param deadpix_file: path to dead pixels file
    :param prefilter_file: path to prefilter file
    :param distortion_file: path to distortion file
    :param niter: int, number of iterations
    :param double_pass: bool
    :param verbose: bool, verbosity parameter
    :return: None
    '''

    if isinstance(files, str):
        if verbose:
            print('looking for files in folder:', files)
        files = sorted(glob.glob(files + '/*.fits*'))

    if verbose:
        print('found', len(files), 'input files')
        print('first input file is:', files[0])
        print('last input file is:', files[-1])

    if verbose:
        print('reading and preprocessing the data')

    if dark_file is None:
        raise Exception('dark signal file not specified')
    if deadpix_file is None:
        raise Exception('dead pixels file not specified')
    if prefilter_file is None:
        raise Exception('prefilter file not specified')
    if distortion_file is None:
        raise Exception('distortion file not specified')

    if verbose:
        print('dark signal file is:', dark_file)
        print('dead pixels file is:', deadpix_file)
        print('prefilter file is:', prefilter_file)
        print('distortion file is:', distortion_file)

    datas = []
    headers = []

    for i, file in enumerate(files):
        with fits.open(file) as hdul:
            header = hdul[0].header
            data = hdul[0].data

        if i == 0:
            pmp_temperature = int(header['FPMPTSP1'])
            fg_temperature = int(header['FGH_TSP1'])
            dsun_au = header['DSUN_AU']
            contposn = header['CONTPOSN']
            wvlns = read_wavelengths(header)

            if verbose:
                print('distance is:', dsun_au, 'AU')
                print('PMP SP temperature is:', pmp_temperature, 'C')
                print('FG SP temperature is:', fg_temperature, 'C')
                print('continuum position is:', contposn)
                print('wavelengths are:', wvlns, 'A')

        datas += [preprocess(data, header,
                             dark_file=dark_file,
                             deadpix_file=deadpix_file,
                             prefilter_file=prefilter_file,
                             distortion_file=distortion_file,
                             verbose=verbose)]
        headers += [header]
    datas = np.array(datas)
    shifts = np.nan_to_num([get_wv_shift(data, contpos=5, delta_wv=0.069) for data in datas])
    images = datas[:, -1]

    centers = []
    for image in images:
        xc, yc, rsun = find_center(image)
        centers.append([xc, yc])
    centers = np.array(centers)
    cavity = kll(shifts, centers, weights=images.clip(0))

    mask = np.any(images > np.max(images) * 0.1, axis=0)
    cavity[~mask] = 0

    if verbose:
        print('done')

    return cavity


def preprocess(data, header,
               dark_file=None,
               prefilter_file=None,
               deadpix_file=None,
               distortion_file=None,
               verbose=True):

    nx, ny = data.shape[-2:]
    cpos = int(header['CONTPOS']) - 1

    if dark_file is not None:
        with fits.open(dark_file) as hdul:
            dark = hdul[0].data
        dark = crop(dark, header)
        data -= 0.4 * crop(dark, header)  ###

    data = np.mean(data.reshape(6, -1, nx, ny), axis=1)

    if prefilter_file is not None:
        data = correct_prefilter(data, header, prefilter_file)

    if cpos == 0:
        data = np.roll(data, -1, axis=0)

    if deadpix_file is not None:
        with fits.open(deadpix_file) as hdul:
            deadpix = hdul[0].data[:, ::-1].astype(bool)
        deadpix = crop(deadpix, header)
        data[...,~deadpix] = np.nan
        data = fill_holes(data)
        data = np.nan_to_num(data)

    if distortion_file is not None:
        s = np.load(distortion_file)
        xd, yd = s['xd'], s['yd']
        data = undistort(data, header, xd, yd)

    return data.astype(np.float32)




