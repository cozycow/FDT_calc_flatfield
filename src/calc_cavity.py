from os import path
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter, binary_dilation
from crosstalk_correction import calc_continuum
from ghost_correction import reflect
from classical_estimates import get_wv_shift
from wavelengths import read_wavelengths
from kll import kll
from fitting import polyfit2d
from limb_fitting import *
from processing import *
from modulation import *


def calc_cavity(files, folder_out='',
                   dark_file=None,
                   deadpix_file=None,
                   prefilter_file=None,
                   flatfield_file=None,
                   ghost_file=None,
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
    elif verbose:
        print('dark signal file is:', dark_file)

    if deadpix_file is None:
        raise Exception('dead pixels file not specified')
    elif verbose:
        print('dead pixels file is:', deadpix_file)

    if prefilter_file is None:
        raise Exception('prefilter file not specified')
    elif verbose:
        print('prefilter file is:', prefilter_file)

    if flatfield_file is None:
        raise Exception('flatfield file not specified')
    elif verbose:
        print('flatfield file is:', flatfield_file)

    if ghost_file is None:
        raise Exception('ghost file not specified')
    elif verbose:
        print('ghost file is:', ghost_file)

    if distortion_file is None:
        raise Exception('distortion file not specified')
    elif verbose:
        print('distortion file is:', distortion_file)

    vlcps = []
    vrcps = []
    weights = []
    centers = []

    for i, file in enumerate(files):
        data, header = process(file,
                                dark_file=dark_file,
                                deadpix_file=deadpix_file,
                                prefilter_file=prefilter_file,
                                flatfield_file=flatfield_file,
                                ghost_file=ghost_file,
                                distortion_file=distortion_file,
                                _realign=True,
                                _find_center=True,
                                _demodulate=True,
                                _correct_fringes=True,
                                _correct_crosstalk=True,
                                #_mask=True,
                                )

        if i == 0:
            header_ = header
            pmp_temperature = int(header['FPMPTSP1'])
            fg_temperature = int(header['FGH_TSP1'])
            dsun_au = header['DSUN_AU']
            contposn = header['CONTPOSN']

            if verbose:
                print('distance is:', dsun_au, 'AU')
                print('PMP SP temperature is:', pmp_temperature, 'C')
                print('FG SP temperature is:', fg_temperature, 'C')
                print('continuum position is:', contposn)

        contpos = header['CONTPOS'] - 1

        xc = header['CRPIX2'] - 1
        yc = header['CRPIX1'] - 1

        lcp = (data[:, 0] + data[:, 3]) / 2
        rcp = (data[:, 0] - data[:, 3]) / 2

        vlcp = get_wv_shift(lcp, header)
        vrcp = get_wv_shift(rcp, header)

        vlcps += [vlcp]
        vrcps += [vrcp]
        weights += [data[contpos, 0]]
        centers += [(xc, yc)]

        if verbose:
            print('file', file, 'processed')

    vlcps = np.array(vlcps)
    vrcps = np.array(vrcps)
    weights = np.array(weights)
    centers = np.array(centers)

    if verbose:
        print('disk centers are:', centers)

    if verbose:
        print('calculating mask')

    mask = np.all(weights < 1000, axis=0)
    mask = binary_dilation(mask, iterations=3)

    if verbose:
        print('calculating cavity')

    cavity_lcp = kll(np.nan_to_num(vlcps), centers, weights=np.nan_to_num(weights),
                     niter=niter, sigma=1e-3, vmin=-0.2, vmax=0.2)
    cavity_rcp = kll(np.nan_to_num(vrcps), centers, weights=np.nan_to_num(weights),
                     niter=niter, sigma=1e-3, vmin=-0.2, vmax=0.2)

    cavity = np.array([cavity_lcp, cavity_rcp])
    cavity[:,mask] = np.nan
    cavity -= np.nanmedian(cavity[:, 512:1536, 512:1536])
    cavity = np.nan_to_num(cavity)

    if verbose:
        print('distorting cavity')

    s = np.load(distortion_file)
    xu, yu = s['xu'], s['yu']

    cavity = undistort(cavity, header_, xu, yu)

    if verbose:
        print('saving result')

    cavity_file = path.join(folder_out, generate_filename(files[0], 'cavity'))
    clone_fits(files[0], cavity_file, cavity)

    if verbose:
        print('cavity map saved to file:', cavity_file)

    if verbose:
        print('done')
