import numpy as np


def correct_prefilter(data, header, prefilter_file, temperature_constant=0.03, cavity_map=None, **kwargs):
    '''
    :param data: numpy array of shape (24,nx,ny) or (6,4,nx,ny)
    :param header: fits header
    :param prefilter_file: path to prefilter file containing 2d-polynomial fit coefficients
    :param cavity_map: cavity map (optional)
    :param temperature_constant: prefilter temperature constant, A/K
    :return: corrected data of same shape
    '''

    if 'temperature' in kwargs:
        temperature = kwargs['temperature']
    elif 'FGOV1PT1' in header:
        temperature = header['FGOV1PT1']
    else:
        raise ValueError('FG temperature not provided')

    if 'wavelengths' in kwargs:
        wavelengths = kwargs['wavelengths']
    elif 'WAVENUM' in header:
        nwv = header['WAVENUM']
        wavelengths = []
        for i in range(nwv):
            wavelengths.append(header[f'WAVELN{i + 1:02d}'])
        wavelengths = np.array(wavelengths)
    else:
        raise ValueError('Wavelengths not provided')

    nx, ny = data.shape[-2:]
    nwv = len(wavelengths)

    prefilter_wavelengths, coefficients, prefilter_temperature = read_prefilter(prefilter_file)
    delta_wv = (temperature - prefilter_temperature) * temperature_constant
    prefilter_wavelengths += delta_wv

    prefilter = interpolate_prefilter(coefficients, prefilter_wavelengths, wavelengths, cavity=cavity_map, header=header)
    data_ = data.copy().reshape((nwv, -1, nx, ny))
    data_ /= np.expand_dims(prefilter, 1)
    return data_.reshape(data.shape)


def read_prefilter(file):
    wv, p = [], []
    T = 61.

    with open(file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.strip():
                if line.startswith('#'):
                    line = line.strip(' #').split(' ')
                    if line[0] == 'Temperature':
                        T = float(line[1])
                else:
                    line = line.split(', ')
                    wv += [line[0]]
                    p += [line[1:]]

    wv = np.array(wv[1:]).astype(np.float32)
    p = np.array(p[1:]).astype(np.float32)
    return wv, p, T


def polyterms2d(x, y, degree=1):
    if degree == 0:
        return [np.ones_like(x)]
    else:
        return np.append(polyterms2d(x, y, degree=degree-1), np.array([x ** (degree - i) * y ** i for i in range(degree + 1)]), axis=0)


def polyval2d(x, y, p):
    degree = int((np.sqrt(8 * len(p) + 1) - 3) / 2)
    return np.sum([p_ * x_ for p_, x_ in zip(p, polyterms2d(x, y, degree=degree))], axis=0)


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image


def interpolate(f, x, x_new):
    idx = np.searchsorted(x, x_new).clip(1, len(x) - 1)
    xa, xb = x[idx - 1], x[idx]
    dx = xb - xa

    a, b = (xb - x_new) / dx, (x_new - xa) / dx
    fa = np.take_along_axis(f, idx - 1, axis=0)
    fb = np.take_along_axis(f, idx, axis=0)
    return fa * a + fb * b


def interpolate_prefilter(coeff, wv_prefilter, wv_data, nx=2048, ny=2048, cavity=None, header=None):
    if cavity is None:
        cavity = np.zeros((nx, ny), dtype=np.float32)

    x, y = np.mgrid[-nx // 2 + 0.5:nx // 2 + 0.5, -ny // 2 + 0.5:ny // 2 + 0.5].astype(np.float32)
    x /= nx / 2
    y /= ny / 2

    x = crop(x, header=header)
    y = crop(y, header=header)

    prefilter = []
    for wv in wv_data:
        p = interpolate(np.expand_dims(coeff, (-1,-2)), wv_prefilter, np.expand_dims(wv - crop(cavity, header=header), (0,1)))[0]
        q = polyval2d(x, y, p)
        prefilter.append(q)
    return np.array(prefilter)
