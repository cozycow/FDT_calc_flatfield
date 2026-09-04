from os import path
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter, binary_dilation
from crosstalk_correction import calc_continuum
from ghost_correction import reflect
from classical_estimates import *
from wavelengths import read_wavelengths
from kll import kll
from fitting import polyfit2d
from limb_fitting import *
from processing import *
from modulation import *


def calc_bias(files, folder_out='',
                   dark_file=None,
                   deadpix_file=None,
                   prefilter_file=None,
                   distortion_file=None,
                   niter=100,
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
    shifts = []
    centers = []

    for i, file in enumerate(files):
        data, header = process(file,
                               dark_file=dark_file,
                               deadpix_file=deadpix_file,
                               prefilter_file=prefilter_file,
                               distortion_file=distortion_file,
                               _find_center=True,
                               verbose=verbose)

        if i == 0:
            header_ = header
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

        xc = header['CRPIX2'] - 1
        yc = header['CRPIX1'] - 1

        blos, vlos = classical_estimates(data, header)
        shifts += [np.nan_to_num(blos)]
        datas += [np.nan_to_num(calc_continuum(data, header))]
        centers += [(xc, yc)]

    datas = np.array(datas)
    shifts = np.array(shifts)
    centers = np.array(centers)

    if verbose:
        print('disk centers are:', centers)

    if verbose:
        print('calculating mask')

    mask = np.all(datas[:,0] < np.max(datas[:,0]) * 0.1, axis=0)
    mask = binary_dilation(mask, iterations=3)

    if verbose:
        print('calculating cavity')

    cavity = kll(shifts, centers, datas[:,0].clip(0),
                 niter=niter, sigma=10, vmin=-300, vmax=300)
    cavity[mask] = np.nan
    cavity -= np.nanmedian(cavity[512:1536, 512:1536])
    cavity = np.nan_to_num(cavity)


    if verbose:
        print('distorting flatfield')

    s = np.load(distortion_file)
    xu, yu = s['xu'], s['yu']

    cavity = undistort(cavity, header_, xu, yu)
    cavity_file = path.join(folder_out, generate_filename(files[0], 'cavity'))
    clone_fits(files[0], cavity_file, cavity)

    if verbose:
        print('done')
