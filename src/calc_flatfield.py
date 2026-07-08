import numpy as np
from astropy.io import fits
import glob
from scipy.ndimage import gaussian_filter

from prefilter_correction import correct_prefilter
from limb_fitting import *
from utils import *


def calc_flatfield(folder_in, folder_out='',
                   dark_file='', deadpix_file='', prefilter_file='', distortion_file='',
                   verbose=True):
    if verbose:
        print('start processing folder', folder_in)

    files = sorted(glob.glob(folder_in + '/*.fits.gz'))

    if verbose:
        print('found', len(files), 'files')

    if verbose:
        print('reading and preprocessing the data')

    datas = []
    headers = []

    for file in files:
        with fits.open(file) as hdul:
            header = hdul[0].header
            data = hdul[0].data

        datas += [preprocess(data, header,
                             dark_file=dark_file,
                             deadpix_file=deadpix_file,
                             prefilter_file=prefilter_file,
                             distortion_file=distortion_file,
                             verbose=verbose)]
        headers += [header]
    datas = np.array(datas)

    temperature = int(headers[0]['FPMPTSP1'])
    if verbose:
        print('the PMP SP temperature is:', temperature)
        print('the FG SP temperature is:', headers[0]['FGH_TSP1'])
        print('the continuum position is:', headers[0]['CONTPOSN'])

    if verbose:
        print('calculating and correcting transmittance')

    transmittance = calc_transmittance(datas[:, 2])
    datas /= np.nan_to_num(transmittance, nan=1)

    if verbose:
        print('realigning and demodulating the data')
        print('modulation matrix is:')
        print(modulation_matrix(temperature))

    for i in range(len(datas)):
        datas[i] = realign(datas[i])
        datas[i] = demodulate(datas[i], temperature=temperature)

    if verbose:
        print('calculating ghost reflection center')

    xr, yr = calc_reflection_center(datas[:, 0], datas[:, 1])

    if verbose:
        print('the reflection center is:', xr, yr)

    if verbose:
        print('calculating instrumental polarization')

    flats, ghosts = [], []
    for i in range(1, 4):
        flat, ghost = calc_polarization(datas[:, 0], datas[:, i], xr, yr)
        flats += [flat]
        ghosts += [ghost]

    flats = np.array(flats)
    ghosts = np.array(ghosts)

    if verbose:
        print('removing fringes')

    flats = remove_fringes(flats)

    if verbose:
        print('modulating flatfield')

    norm = modulation_matrix(temperature)[:, 0]

    flats = np.append(np.ones((1, 2048, 2048)), flats, axis=0) * transmittance
    ghosts = np.append(np.zeros((1, 2048, 2048)), ghosts, axis=0)

    flats = modulate(flats, temperature=temperature) / norm.reshape(-1, 1, 1)
    ghosts = modulate(ghosts, temperature=temperature)

    if verbose:
        print('distorting flatfield')

    s = np.load(distortion_file)
    xu, yu = s['xu'], s['yu']

    flats = np.nan_to_num(flats, nan=1)
    ghosts = np.nan_to_num(ghosts, nan=0)

    flats = undistort(flats, headers[0], xu, yu, cval=1)
    ghosts = undistort(ghosts, headers[0], xu, yu)

    if verbose:
        print('saving the result')

    flat_file = folder_out + '/' + generate_filename(files[0], 'flat')
    ghost_file = folder_out + '/' + generate_filename(files[0], 'ghost')

    clone_fits(files[0], flat_file, flats)
    clone_fits(files[0], ghost_file, ghosts)

    if verbose:
        print('the output flatfield file is:', flat_file)
        print('the output ghost file is:', ghost_file)

    if verbose:
        print('done')

    return flat_file, ghost_file


def preprocess(data, header,
               dark_file='', deadpix_file='', prefilter_file='', distortion_file='', verbose=True):

    with fits.open(dark_file) as hdul:
        dark = hdul[0].data

    with fits.open(deadpix_file) as hdul:
        deadpix = hdul[0].data[:,::-1].astype(bool)

    s = np.load(distortion_file)
    xd, yd = s['xd'], s['yd']

    wv = read_wavelengths(header)
    cpos = int(header['CONTPOS']) - 1

    data -= 0.4 * dark ###
    data = correct_prefilter(data, header, prefilter_file)
    data = calc_continuum(data, wv, continuum=cpos)
    data[:,~deadpix] = np.nan
    data = fill_holes(data)
    data = np.nan_to_num(data)
    data = undistort(data, header, xd, yd)

    return data.astype(np.float32)


def calc_transmittance(images, niter=10):

    #calculating disk centers
    centers = []
    for image in images:
        xc, yc, rsun = find_center(image)
        centers.append(np.array([xc, yc]))

    #calculating intensity threshold
    a = np.percentile(images[0], 0.1)
    b = np.percentile(images[0], 99.9)
    threshold = a + (b - a) * 0.1

    flatfield = np.ones_like(images[0])
    for _ in range(niter):

        #calculating average image
        mean_image = np.zeros_like(images[0])
        coverage = np.zeros_like(images[0])
        for image, center in zip(images, centers):
            with np.errstate(invalid='ignore'):
                image_ = image / np.nan_to_num(flatfield, nan=1)

            image_ = roll_float(image_, *(centers[0] - center))

            weight = image_.copy()
            weight[weight < threshold] = 0
            coverage += weight
            mean_image += (image_ - mean_image) * weight / coverage.clip(1)

        A = np.zeros_like(images[0])
        B = np.zeros_like(images[0])
        coverage = np.zeros_like(images[0])
        for image, center in zip(images, centers):
            image_ = roll_float(mean_image, *(center - centers[0]))

            weight = image.copy()
            weight[weight < threshold] = 0
            coverage += weight

            A += (image_ * image - A) * weight / coverage.clip(1)
            B += (image_ ** 2 - B) * weight / coverage.clip(1)

        with np.errstate(invalid='ignore'):
            flatfield = A / B

    return flatfield.astype(np.float32)


def calc_polarization(I, Q, xr, yr, degree=2, sigma=30, niter=3):

    a = np.percentile(I[0], 0.1)
    b = np.percentile(I[0], 99.9)
    threshold = a + (b - a) * 0.1

    I_ = reflect(I, xr, yr)
    I_ = gaussian_filter(I_, 8, axes=(-2,-1))

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
    from scipy.ndimage import binary_dilation
    from skimage.feature import canny

    a = np.percentile(I[0], 0.1)
    b = np.percentile(I[0], 99.9)
    threshold = a + (b - a) * 0.1

    X, Y = [], []

    for i in range(len(I)):
        mask = binary_dilation(I[i] > threshold, iterations=5)
        edges = canny(Q[i], sigma=8, low_threshold=0.9, high_threshold=0.9, use_quantiles=True)
        edges *= ~mask

        xe, ye = np.where(edges)
        xe, ye = filter_outliers(xe, ye)

        xg, yg, rg = fitnp(xe, ye)
        xc, yc, rs = find_center(I[i])

        X.append((xc + xg) / 2)
        Y.append((yc + yg) / 2)

    return np.median(X), np.median(Y)


def remove_fringes(data, sigma=0.01, degree=7):
    if len(data.shape) == 2:
        temp = data.copy()
        fit = polyfit2d(temp.clip(-sigma, sigma), degree=degree)
        temp = temp - fit
        temp = remove_freq(temp, (3, 15, 18, 22), (27, 27, 20, 16), h=5, thr=sigma, fill=0)
        temp = remove_freq(temp, (12, 4), (0, -4), h=1, thr=sigma, fill=0)
        return temp + fit
    else:
        return np.array([remove_fringes(temp, sigma=sigma, degree=degree) for temp in data])

