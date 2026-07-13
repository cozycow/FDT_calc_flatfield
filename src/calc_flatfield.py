import numpy as np
from astropy.io import fits
import glob
from scipy.ndimage import gaussian_filter, binary_dilation

from prefilter_correction import correct_prefilter
from limb_fitting import *
from utils import *


def calc_flatfield(files, folder_out='',
                   dark_file=None,
                   deadpix_file=None,
                   prefilter_file=None,
                   distortion_file=None,
                   niter=10,
                   true_continuum=True,
                   double_pass=True,
                   #save_fringes=False,
                   verbose=True):

    '''
    :param files: list of input files paths or path to input files folder
    :param folder_out: output folder path
    :param dark_file: path to dark signal file
    :param deadpix_file: path to dead pixels file
    :param prefilter_file: path to prefilter file
    :param distortion_file: path to distortion file
    :param niter: int, number of iterations
    :param true_continuum: bool, whether to filter out ARs or not
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
                             true_continuum=true_continuum,
                             verbose=verbose)]
        headers += [header]
    datas = np.array(datas)

    if verbose:
        print('calculating transmittance')

    transmittance = calc_transmittance(datas[:, 0], niter=niter)

    if verbose:
        print('correcting data for transmittance')

    datas /= np.nan_to_num(transmittance, nan=1)

    if verbose:
        print('realigning and demodulating the data')
        print('modulation matrix is:')
        print(modulation_matrix(pmp_temperature))

    for i in range(len(datas)):
        datas[i] = realign(datas[i])
        datas[i] = demodulate(datas[i], temperature=pmp_temperature)

    if verbose:
        print('calculating ghost reflection center')

    xr, yr = calc_reflection_center(datas[:, 0], datas[:, 1])

    if verbose:
        print('reflection center is:', xr, yr)

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
    #fringes = flats - remove_fringes(flats)
    #flats -= fringes

    flats = np.append(np.ones((1, 2048, 2048)), flats, axis=0)
    #fringes = np.append(np.zeros((1, 2048, 2048)), fringes, axis=0)
    ghosts = np.append(np.linalg.norm(ghosts, axis=0, keepdims=True) * 3, ghosts, axis=0) ###

    if double_pass:
        if verbose:
            print('removing ghosts from data and recalculating transmittance')

        datas[:,0] -= reflect(gaussian_filter(datas[:,0], 8, axes=(-2,-1)), xr, yr) * ghosts[0]
        transmittance *= calc_transmittance(datas[:, 0], niter=niter)

    if verbose:
        print('normalizing transmittance')

    transmittance /= np.nanmedian(transmittance[512:1536, 512:1536])

    if verbose:
        print('modulating flatfield')

    norm = modulation_matrix(pmp_temperature)[:, 0]
    flats = modulate(flats, temperature=pmp_temperature) / norm.reshape(-1, 1, 1)
    #fringes = modulate(fringes, temperature=pmp_temperature)
    ghosts = modulate(ghosts, temperature=pmp_temperature)
    flats *= transmittance

    if verbose:
        print('filling missing values')

    mask = np.isnan(flats[0])
    mask = binary_dilation(mask, iterations=3)

    flats[:,mask] = 1.
    #fringes[:,mask] = 0.
    #ghosts[:,mask] = 0.

    if verbose:
        print('distorting flatfield')

    s = np.load(distortion_file)
    xu, yu = s['xu'], s['yu']

    flats = undistort(flats, headers[0], xu, yu, cval=1)
    #fringes = undistort(fringes, headers[0], xu, yu)
    ghosts = undistort(ghosts, headers[0], xu, yu)

    if verbose:
        print('clipping flatfield')

    flats = flats.clip(0.1, 2)

    if verbose:
        print('saving result')

    flat_file = folder_out + '/' + generate_filename(files[0], 'flat')
    ghost_file = folder_out + '/' + generate_filename(files[0], 'ghost')
    #fringes_file = folder_out + '/' + generate_filename(files[0], 'fringes')

    clone_fits(files[0], flat_file, flats)

    if verbose:
        print('flatfield map saved to file:', flat_file)

    clone_fits(files[0], ghost_file, ghosts)

    if verbose:
        print('ghost map saved to file:', ghost_file)

    #if save_fringes:
    #    clone_fits(files[0], fringes_file, fringes)

    #    if verbose:
    #        print('fringes map saved to file:', fringes_file)

    if verbose:
        print('done')

    #return datas


def preprocess(data, header,
               dark_file=None,
               deadpix_file=None,
               prefilter_file=None,
               distortion_file=None,
               true_continuum=True,
               verbose=True):

    with fits.open(dark_file) as hdul:
        dark = hdul[0].data

    with fits.open(deadpix_file) as hdul:
        deadpix = hdul[0].data[:,::-1].astype(bool)

    dark = crop(dark, header)
    deadpix = crop(deadpix, header)

    s = np.load(distortion_file)
    xd, yd = s['xd'], s['yd']

    wv = read_wavelengths(header)
    cpos = int(header['CONTPOS']) - 1

    data -= 0.4 * dark ###
    data = correct_prefilter(data, header, prefilter_file)

    if true_continuum:
        data = calc_continuum(data, wv, continuum=cpos)
    else:
        data = data.reshape(6, 4, 2048, 2048)[cpos]

    data[:,~deadpix] = np.nan
    data = fill_holes(data)
    data = np.nan_to_num(data)
    data = undistort(data, header, xd, yd)

    return data.astype(np.float32)


def calc_transmittance(images, niter=20):

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
    if len(data.shape) == 2:
        temp = data.copy()
        fit = polyfit2d(temp.clip(-sigma, sigma), degree=degree)
        temp = temp - fit
        temp = remove_freq(temp, (3, 15, 18, 22), (27, 27, 20, 16), h=5, thr=sigma, fill=0)
        temp = remove_freq(temp, (12, 4), (0, -4), h=1, thr=sigma, fill=0)
        return temp + fit
    else:
        return np.array([remove_fringes(temp, sigma=sigma, degree=degree) for temp in data])

