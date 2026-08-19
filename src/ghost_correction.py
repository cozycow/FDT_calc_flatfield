import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def correct_ghost(data, header, ghost, **kwargs):
    '''
    :param data: numpy array of shape (24,nx,ny) or (6,4,nx,ny) containing modulated intensities
    :param header: fits header
    :param ghost: ghost map of shape (4,nx,ny)
    :return: corrected data
    '''

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

    xr, yr = reflection_point_predict(header)

    data_ = data.copy().reshape((6, -1, data.shape[-2], data.shape[-1]))
    ghost_matrix = generate_ghost_matrix(wavelengths, contpos=contpos, **kwargs)

    reflection = data_[:,:1].copy()
    reflection = gaussian_filter(reflection, 8, axes=(-2, -1))
    reflection = reflect(reflection, xr, yr)
    reflection = np.matmul(ghost_matrix, reflection, axes=[(-2, -1), (0, 1), (0, 1)])
    data_ -= reflection * np.expand_dims(crop(ghost, header), 0)

    return data_.reshape(data.shape)


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image


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


def voigt_profile(z, sigma, gamma, modified=False, acc=1e-3, truncate=4):
    if isinstance(z, np.ndarray):
        return np.array([voigt_profile(z_, sigma, gamma, modified=modified, acc=acc, truncate=truncate) for z_ in z])
    else:
        limit = truncate * (sigma + gamma)
        x = np.arange(-limit,limit + acc / 2, acc)
        f = np.exp(-x ** 2 / 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))
        g = 1 / (1 + ((z - x) / gamma) ** 2)

        if modified:
            g = g * (1 - g) * 2 / (gamma * np.pi)
        else:
            g = g / (gamma * np.pi)

        return np.trapezoid(f * g, x)


def generate_ghost_matrix(wv, sigma=0.043, gamma=0.053, contpos=-1, acc=1e-3, lam=1e-6, **kwargs):
    n_wv = len(wv)

    wv_min = np.min(np.delete(wv, contpos) if contpos is not None else wv)
    wv_max = np.max(np.delete(wv, contpos) if contpos is not None else wv)
    wvc = (wv_min + wv_max) / 2
    wv_ = np.arange(wv_min, wv_max + acc / 2, acc, dtype=np.float32)
    n_wv_ = len(wv_)

    xi = np.expand_dims(wv, axis=1) - np.expand_dims(wv_, axis=0)
    A = voigt_profile(xi, sigma, gamma)
    A0 = np.mean(A, axis=0, keepdims=True)
    A -= A0

    M = np.zeros((n_wv, n_wv))
    N = np.zeros((n_wv, n_wv))

    for k in range(n_wv_):
        gk = voigt_profile(wv_[k] - wvc, sigma, gamma) ** 2

        for j in range(n_wv):
            mjk = gk * (voigt_profile(wv_[k] - wv[j], sigma, gamma, modified=True) - A0[0, k])
            njk = gk * A[j, k]

            for i in range(n_wv):
                M[j, i] += mjk * A[i, k]
                N[j, i] += njk * A[i, k]

    Q = M @ np.linalg.inv(N + lam * np.identity(n_wv))
    return Q @ (np.identity(n_wv) - 1 / n_wv) + 1 / n_wv


def reflection_point_predict(header):
    px = [1.63114715e-06, 6.72511045e-03, 9.60448053e+02]
    py = [ 4.61830880e-06, -6.85005911e-03,  9.77508840e+02]

    r_sun = header['RSUN_ARC']
    dx, dy = header['PXBEG2'] - 1, header['PXBEG1'] - 1

    xr = np.polyval(px, r_sun) - dx
    yr = np.polyval(py, r_sun) - dy
    return xr, yr
