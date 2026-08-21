import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates
from skimage.restoration import inpaint
from scipy.ndimage import binary_fill_holes


def read_wavelengths(header):
    nwv = header['WAVENUM']
    wv = []
    for i in range(nwv):
        wv.append(header[f'WAVELN{i + 1:02d}'])
    return np.array(wv)


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image


def rebin(data, k, axis=None):
    if len(data.shape) == 2:
        nx, ny = data.shape
        if axis == 0:
            return np.mean(np.reshape(data[:nx // k * k, :], (nx // k, -1, ny)), axis=-2)
        elif axis == 1:
            return np.mean(np.reshape(data[:, :ny // k * k], (nx, ny // k, -1)), axis=-1)
        else:
            return rebin(rebin(data, k, axis=0), k, axis=1)
    else:
        out = []
        for i in range(len(data)):
            out.append(rebin(data[i], k, axis=axis))
        return np.array(out)


def roll_float(data, dx, dy, **kwargs):
    if len(data.shape) == 2:
        nx, ny = data.shape
        xi, yi = np.mgrid[:nx,:ny].astype(np.float32)
        xi -= dx
        yi -= dy
        return map_coordinates(data, (xi, yi), **kwargs)
    else:
        out = []
        for i in range(len(data)):
            out.append(roll_float(data[i], dx, dy, **kwargs))
        return np.array(out)


def reflect(data, xr, yr, **kwargs):
    nx, ny = data.shape[-2:]
    return roll_float(data[...,::-1, ::-1], 2 * int(round(xr)) - nx + 1, 2 * int(round(yr)) - ny + 1, **kwargs)


def fill_holes(data):
    if len(data.shape) == 2:
        mask = np.isnan(data)
        not_holes = binary_fill_holes(~mask)
        holes = not_holes & mask
        image_ = np.nan_to_num(data, nan=0)
        image_ = inpaint.inpaint_biharmonic(image_, holes)
        image_[~not_holes] = np.nan
        return image_
    else:
        out = []
        for i in range(len(data)):
            out.append(fill_holes(data[i]))
        return np.array(out)


def polyterms2d(x, y, degree=1):
    if degree == 0:
        return np.array([np.ones_like(x)])
    else:
        return np.append(polyterms2d(x, y, degree=degree-1), np.array([x ** (degree - i) * y ** i for i in range(degree + 1)]), axis=0)


def polyval2d(x, y, p):
    degree = int((np.sqrt(8 * len(p) + 1) - 3) / 2)
    return np.sum([p_ * x_ for p_, x_ in zip(p, polyterms2d(x, y, degree=degree))], axis=0)


def polyfit2d(f, x=None, y=None, degree=1, weight=None, return_coefficients=False):
    if x is None and y is None:
        nx, ny = f.shape
        x, y = np.mgrid[-nx // 2 + 0.5:nx // 2 + 0.5, -ny // 2 + 0.5:ny // 2 + 0.5].astype(np.float32)
        x /= nx / 2
        y /= ny / 2

    if weight is None:
        W = np.ones_like(f)
    else:
        W = weight

    t = np.where(~np.isnan(f))
    X = polyterms2d(x[t], y[t], degree=degree)[1:]
    Y = f[t]
    W = W[t]
    W /= np.sum(W)

    X0 = np.sum(X * W, axis=-1, keepdims=True)
    Y0 = np.sum(Y * W)

    X_ = X - X0
    Y_ = Y - Y0

    k = (Y_ * W) @ X_.T @ np.linalg.inv((X_ * W) @ X_.T)
    b = Y0 - k @ X0
    p = np.append(b, k)

    if return_coefficients:
        return p
    else:
        return polyval2d(x, y, p)


def interpolate(f, x, x_new):
    idx = np.searchsorted(x, x_new).clip(1, len(x) - 1)
    xa, xb = x[idx - 1], x[idx]
    dx = xb - xa

    a, b = (xb - x_new) / dx, (x_new - xa) / dx
    fa = np.take_along_axis(f, idx - 1, axis=0)
    fb = np.take_along_axis(f, idx, axis=0)
    return fa * a + fb * b


def get_wv_shift(data, header, pol=0, log=True, **kwargs):
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

    if 'contpos' in kwargs:
        contpos = kwargs['contpos']
    elif 'CONTPOS' in header:
        contpos = header['CONTPOS'] - 1
    else:
        raise ValueError('Continuum position not provided')

    wavelengths = np.delete(wavelengths, contpos)
    delta_wv = np.mean(wavelengths[1:] - wavelengths[:-1])

    temp = data.copy().reshape((6, -1, data.shape[-2], data.shape[-1]))[:, pol]

    if log:
        temp = -np.log((temp[contpos] - np.delete(temp, contpos, axis=0)).clip(1))
    else:
        temp = np.delete(temp, contpos, axis=0)

    t = np.argmin(temp, axis=0)
    l, a, r = np.take_along_axis(temp, np.array([(t - 1) % 5, t, (t + 1) % 5]), axis=0)
    b, c = (r - l) / 2, (l + r) - 2 * a

    with np.errstate(invalid='ignore'):
        return np.nan_to_num(t - 2 - b / c) * delta_wv
