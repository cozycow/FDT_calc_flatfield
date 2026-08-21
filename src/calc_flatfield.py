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


def calc_flatfield(files, folder_out='',
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

        shifts += [get_wv_shift(data, header)]
        datas += [calc_continuum(data, header)]
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
                 niter=niter, sigma=1e-3, vmin=-0.2, vmax=0.2)
    cavity[mask] = np.nan
    cavity -= np.nanmedian(cavity[512:1536, 512:1536])
    cavity = np.nan_to_num(cavity)

    if verbose:
        print('calculating transmittance')

    transmittance = kll(datas[:,0], centers, datas[:,0].clip(0),
                        niter=niter, sigma=100, slope=True, vmin=0.1, vmax=2)
    transmittance[mask] = 1

    if verbose:
        print('correcting data for transmittance')

    datas /= transmittance

    if verbose:
        print('realigning and demodulating the data')
        print('modulation matrix is:')
        print(modulation_matrix(temperature=pmp_temperature))

    for i in range(len(datas)):
        datas[i] = realign(datas[i])
        datas[i] = demodulate(datas[i], header_)

    if verbose:
        print('calculating ghost reflection center')

    xr, yr = calc_reflection_center(datas[:, 0], datas[:, 1])

    if verbose:
        print('reflection center is:', xr, yr)

    if verbose:
        print('calculating instrumental polarization')

    flats, ghosts = [], []
    for i in range(1, 4):
        flat, ghost = calc_polarization(datas[:, 0], datas[:, i], xr, yr, niter=niter)
        flats += [flat]
        ghosts += [ghost]

    flats = np.array(flats)
    ghosts = np.array(ghosts)

    if verbose:
        print('removing fringes')

    flats = remove_fringes(flats)

    flats = np.append(np.ones((1, 2048, 2048)), flats, axis=0)
    ghosts = np.append(np.linalg.norm(ghosts, axis=0, keepdims=True) * 3, ghosts, axis=0) ###

    if verbose:
        print('removing ghosts from data and recalculating transmittance')

    datas[:,0] -= reflect(gaussian_filter(datas[:,0], 8, axes=(-2,-1)), xr, yr) * ghosts[0]
    transmittance *= kll(datas[:,0], centers, datas[:,0].clip(0),
                    niter=niter, sigma=100, slope=True, vmin=0.1, vmax=2)
    transmittance[mask] = np.nan

    if verbose:
        print('normalizing transmittance')

    transmittance_norm = np.nanmedian(transmittance[512:1536, 512:1536])

    if verbose:
        print('transmittance norm is:', transmittance_norm)

    transmittance /= transmittance_norm

    if verbose:
        print('modulating flatfield')

    norm = modulation_matrix(temperature=pmp_temperature)[:, 0]
    flats = modulate(flats, header_) / norm.reshape(-1, 1, 1)
    ghosts = modulate(ghosts, header_)
    flats *= transmittance

    if verbose:
        print('filling missing values')

    flats = np.nan_to_num(flats, nan=1.)
    flats = flats.clip(0.1, 2)

    if verbose:
        print('distorting flatfield')

    s = np.load(distortion_file)
    xu, yu = s['xu'], s['yu']

    flats = undistort(flats, header_, xu, yu, cval=1)
    ghosts = undistort(ghosts, header_, xu, yu)
    cavity = undistort(cavity, header_, xu, yu)

    if verbose:
        print('saving result')

    flat_file = path.join(folder_out, generate_filename(files[0], 'flat'))
    ghost_file = path.join(folder_out, generate_filename(files[0], 'ghost'))
    cavity_file = path.join(folder_out, generate_filename(files[0], 'cavity'))
    quicklook_file = path.join(folder_out, generate_filename(files[0], extension='.png'))

    clone_fits(files[0], flat_file, flats)

    if verbose:
        print('flatfield map saved to file:', flat_file)

    clone_fits(files[0], ghost_file, ghosts)

    if verbose:
        print('ghost map saved to file:', ghost_file)

    clone_fits(files[0], cavity_file, cavity)

    if verbose:
        print('cavity map saved to file:', cavity_file)

    if quicklook:
        if verbose:
            print('making quicklook image')

        make_quicklook(files, quicklook_file,
                       dark_file=dark_file,
                       prefilter_file=prefilter_file,
                       deadpix_file=deadpix_file,
                       flatfield_file=flat_file,
                       ghost_file=ghost_file,
                       #_realign=True,
                       _demodulate=True,
                       _correct_fringes=True)

        if verbose:
            print('quicklook image saved to file:', quicklook_file)

    if verbose:
        print('done')


def make_quicklook(files, file_out, **kwargs):
    plt.ioff()

    fig, axs = plt.subplots(4, len(files), figsize=(18,8))

    for i, file in enumerate(files):
        data, header = process(file, **kwargs)
        cpos = int(header['CONTPOS']) - 1
        data = data[cpos]

        a, b = np.nanpercentile(data[0], 0.1), np.nanpercentile(data[0], 99.9)

        axs[0,i].imshow(data[0], origin='lower', cmap='gray', vmin=a, vmax=b)
        axs[1,i].imshow(data[1], origin='lower', cmap='gray', vmin=-1e-3 * (b - a), vmax=1e-3 * (b - a))
        axs[2,i].imshow(data[2], origin='lower', cmap='gray', vmin=-1e-3 * (b - a), vmax=1e-3 * (b - a))
        axs[3,i].imshow(data[3], origin='lower', cmap='gray', vmin=-1e-3 * (b - a), vmax=1e-3 * (b - a))

        axs[0,i].set_title(file.split('_')[-1].split('.')[0])

        for j in range(3):
            axs[j,i].set_xticks([])
            axs[j,i].set_xticklabels([])

        if i == 0:
            axs[0,i].set_ylabel('I')
            axs[1,i].set_ylabel('Q')
            axs[2,i].set_ylabel('U')
            axs[3,i].set_ylabel('V')
        else:
            for j in range(4):
                axs[j,i].set_yticks([])
                axs[j,i].set_yticklabels([])

    plt.tight_layout()
    plt.savefig(file_out)
    plt.close(fig)

    plt.ion()


def calc_polarization(I, Q, xr, yr, degree=2, sigma=5, niter=100):

    a = np.percentile(I[0], 0.1)
    b = np.percentile(I[0], 99.9)
    threshold = a + (b - a) * 0.1

    I_ = reflect(gaussian_filter(I, 8, axes=(-2,-1)), xr, yr)

    a = np.mean(I ** 2, axis=0)
    b = np.mean(I * I_, axis=0)
    d = np.mean(I_ ** 2, axis=0)

    u = np.mean(I * Q, axis=0)
    v = np.mean(I_ * Q, axis=0)

    lam1 = (a + d) / 2 - np.sqrt((a - d) ** 2 / 4 + b ** 2)
    lam2 = (a + d) / 2 + np.sqrt((a - d) ** 2 / 4 + b ** 2)

    mask = np.any(I > threshold, axis=0)
    with np.errstate(invalid='ignore'):
        k = np.abs(lam1 / lam2)
    k[~mask] = 0

    with np.errstate(invalid='ignore'):
        G = (a * v - b * u) / (a * d - b ** 2)

    G[~mask] = np.nan
    G = polyfit2d(G, degree=degree, weight=k)

    W = I
    for _ in range(niter):
        a = np.mean(I * (Q - G * I_) * W, axis=0)
        b = np.mean(I ** 2 * W, axis=0)
        with np.errstate(invalid='ignore'):
            F = a / b
        W = I / np.abs(Q - F * I - G * I_).clip(sigma)

    F[~mask] = np.nan
    return F, G


def calc_reflection_center(I, Q):
    from scipy.ndimage import binary_dilation, binary_erosion
    from skimage.feature import canny

    a = np.percentile(I[0], 0.1)
    b = np.percentile(I[0], 99.9)
    threshold = a + (b - a) * 0.1

    d = np.percentile(np.abs(Q[0]), 99)
    threshold_ = d * 0.04

    X, Y = [], []

    for i in range(len(I)):
        mask = I[i] > threshold
        mask = binary_dilation(mask, iterations=20) * ~binary_erosion(mask, iterations=20)

        edges = canny(Q[i], sigma=8, low_threshold=threshold_, high_threshold=threshold_)
        edges *= ~mask

        xe, ye = np.where(edges)
        xe, ye = filter_outliers(xe, ye)

        xg, yg, rg = fitnp(xe, ye)
        xc, yc, rs = find_center(I[i])

        X.append((xc + xg) / 2)
        Y.append((yc + yg) / 2)

    return np.median(X), np.median(Y)


def remove_fringes(data, sigma=0.01, degree=7):
    from fringe_correction import remove_freq

    if len(data.shape) == 2:
        temp = data.copy()
        fit = polyfit2d(temp.clip(-sigma, sigma), degree=degree)
        temp = temp - fit
        temp = remove_freq(temp, (3, 15, 18, 22), (27, 27, 20, 16), h=5, thr=sigma, fill=0)
        temp = remove_freq(temp, (12, 4), (0, -4), h=1, thr=sigma, fill=0)
        return temp + fit
    else:
        return np.array([remove_fringes(temp, sigma=sigma, degree=degree) for temp in data])

