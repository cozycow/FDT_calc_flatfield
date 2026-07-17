from os import path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from utils import *



def calc_prefilter(files, folder_out='', dark_file='',
                   T0=61, wv0=6173, k_T=0.030, gamma=0.053, delta=0.01, binning=8,
                   verbose=True):
    k_V = 6173.341 / 299792458

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

    if verbose:
        print('dark signal file is:', dark_file)

    with fits.open(dark_file) as hdul:
        dark = hdul[0].data

    datas = []
    wvs = []
    Ts = []
    Vs = []
    for file in files:
        with fits.open(file) as hdul:
            header = hdul[0].header
            data = hdul[0].data

        data = data - 0.4 * dark
        data = rebin(data, binning)
        wv = read_wavelengths(header)

        wvs.append(wv)
        datas.append(data)
        Ts.append(float(header['FGOV1PT1']))
        Vs.append(float(header['OBS_VR']))

    if verbose:
        print('temperatures are:', Ts)
        print('velocities are:', Vs)

    wvs = np.array(wvs)
    datas = np.array(datas)
    Ts = np.array(Ts)
    Vs = np.array(Vs)

    if verbose:
        print('fitting the line profile')

    Shift = np.zeros_like(datas[0][0])
    Sigma = np.zeros_like(datas[0][0])
    Depth = np.zeros_like(datas[0][0])
    Cost = np.zeros_like(datas[0][0])

    for i in range(datas[0].shape[-2]):
        for j in range(datas[0].shape[-1]):
            shift, sigma, depth, cost = fit_line(wvs,datas[...,i,j], Ts, Vs, k_T=k_T, gamma=gamma)

            Shift[i,j] = shift
            Depth[i,j] = depth
            Sigma[i,j] = sigma
            Cost[i,j] = cost

    if verbose:
        print('removing the line and averaging')

    wvs_ = wvs - k_T * (np.expand_dims(Ts, 1) - T0)
    wv_min = np.round(np.min(wvs_), 2)
    wv_max = np.round(np.max(wvs_), 2)
    wv = np.arange(wv_min + delta, wv_max - delta / 2, delta)

    Q = datas / line_profile(np.expand_dims(wvs, (2,3)), np.expand_dims(Shift, (0,1)) + k_V * np.expand_dims(Vs, (1,2,3)), Sigma, gamma, Depth)
    Q = np.array([interpolate(Q[i], wvs_[i], np.expand_dims(wv, (1,2))) for i in range(3)])
    W = np.array([1 - np.cos(2 * np.pi * (wv.clip(wvs_[i,0], wvs_[i,-1]) - wvs_[i,0]) / (wvs_[i,-1] - wvs_[i,0])).reshape(-1,1,1) for i in range(3)])
    Q = np.sum(Q * W, axis=0) / np.sum(W, axis=0)

    if verbose:
        print('normalizing the prefilter')

    Q0 = interpolate(Q, wv, np.expand_dims([wv0], (1,2)))[0]
    Q /= Q0

    if verbose:
        print('fitting the prefilter with 2d-polynomial')

    a = np.percentile(datas, 0.1)
    b = np.percentile(datas, 99.9)
    threshold = a + (b - a) * 0.1
    mask = np.all(datas > threshold, axis=(0,1))
    Q[:,~mask] = np.nan

    P = []
    for i in range(len(Q)):
        Q_ = Q[i].copy()
        Q_[np.abs(Q_ - polyfit2d(Q_, degree=4)) > 0.05] = np.nan
        p = polyfit2d(Q_, degree=4, return_coefficients=True)
        P.append(p)

    P = np.array(P)
    P = gaussian_filter(P, 1, axes=(0,))

    if verbose:
        print('writing prefilter file')

    prefilter_file = path.join(folder_out, generate_filename(files[0], prefix='pref', extension='.txt'))
    with open(prefilter_file, 'w') as file:
        file.write(f'# Temperature {T0}\n')
        file.write(f'# Tuning constant {k_T}\n')
        file.write(f'# Ref wavelength {wv0}\n')
        file.write(f'# Gamma {gamma}\n')
        file.write('\n')
        file.write('wv, p00, p10, p01, p20, p11, p02, p30, p21, p12, p03, p40, p31, p22, p13, p40\n')

        for i in range(len(wv)):
            file.write(f'{wv[i]:.04f}, ' + ', '.join([f'{P[i,j]:.06f}' for j in range(P.shape[-1])]) + '\n')

    if verbose:
        print('done')


def line_profile(x, x0, sigma, gamma, depth):
    from scipy.special import voigt_profile
    f = voigt_profile(x - x0, sigma, gamma)
    f /= voigt_profile(0, sigma, gamma)
    return 1 - depth * f


def residuals(args, wavelengths, profiles, temperatures, velocities, k_T, gamma, wv0):
    k_V = wv0 / 299792458

    shift, sigma, depth = args

    x1, x2, x3 = wavelengths
    y1, y2, y3 = profiles
    T1, T2, T3 = temperatures
    V1, V2, V3 = velocities
    dx1, dx2, dx3 = k_T * T1, k_T * T2, k_T * T3

    f1 = line_profile(x1, shift + k_V * V1, sigma, gamma, depth)
    f2 = line_profile(x2, shift + k_V * V2, sigma, gamma, depth)
    f3 = line_profile(x3, shift + k_V * V2, sigma, gamma, depth)

    y21 = np.interp(x1 - dx1, x2 - dx2, y2 / f2, left=np.nan, right=np.nan)
    y32 = np.interp(x2 - dx2, x3 - dx3, y3 / f3, left=np.nan, right=np.nan)
    y13 = np.interp(x3 - dx3, x1 - dx1, y1 / f1, left=np.nan, right=np.nan)

    z21 = y21 - y1 / f1
    z32 = y32 - y2 / f2
    z13 = y13 - y3 / f3
    return np.nan_to_num(np.append(z21, [z32, z13]))


def fit_line(wavelengths, profiles, temperatures, velocities, k_T=0.030, gamma=0.053, wv0=6173.341):
    result = least_squares(residuals, np.array([wv0, 0.04, 0.5]),
                       bounds=([wv0 - 1, 0.005, 0.01],
                               [wv0 + 1, 0.2, 1]),
                       args=(wavelengths, profiles, temperatures, velocities, k_T, gamma, wv0))

    shift, sigma, depth = result.x
    cost = result.cost
    return shift, sigma, depth, cost
