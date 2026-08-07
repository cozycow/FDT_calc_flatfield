import numpy as np


def correct_cavity(data, header, cavity, **kwargs):
    '''
    :param data: numpy array of shape (24,nx,ny) or (6,4,nx,ny)  containing modulated intensities
    :param header: fits header
    :param cavity: cavity map of shape (nx,ny) ## assuming that it is already cropped
    :return: resampled data
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

    data_ = data.copy().reshape((6, -1, data.shape[-2], data.shape[-1]))
    cavity_ = crop(cavity, header=header)

    wv_shift = cavity_ - get_wv_shift(data_, header, **kwargs)
    offset = np.mean(wv_shift * data_[contpos,0]) / np.mean(data_[contpos,0])
    cavity_matrix = generate_cavity_matrix(cavity_ - offset, wavelengths, contpos=contpos)

    data_ = np.matmul(cavity_matrix, data_, axes=[(-2, -1), (0, 1), (0, 1)])
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


def generate_cavity_matrix(shift, wv, sigma=0.04, gamma=0.05, contpos=-1,
                           fit=True, spread=0.07, acc=1e-3, lam=1e-6, **kwargs):
    from scipy.special import voigt_profile

    if fit:
        dx = spread / 3
        delta = np.arange(-spread, spread + dx / 2, dx)
        M = generate_cavity_matrix(delta, wv, sigma=sigma, gamma=gamma, contpos=contpos,
                                   fit=False, spread=spread, acc=acc, lam=lam, **kwargs)
        P = np.polyfit(delta, M.reshape((len(delta), -1)), 6)
        return np.polyval(P, np.expand_dims(shift.flatten(), axis=1)).reshape(shift.shape + (len(wv), len(wv)))

    n_wv = len(wv)
    shape = shift.shape

    wv_min = np.min(np.delete(wv, contpos) if contpos is not None else wv)
    wv_max = np.max(np.delete(wv, contpos) if contpos is not None else wv)
    wvc = (wv_min + wv_max) / 2
    wv_ = np.arange(wv_min, wv_max + acc / 2, acc, dtype=np.float32)
    n_wv_ = len(wv_)

    xi = np.expand_dims(wv, axis=1) - np.expand_dims(wv_, axis=0)
    A = voigt_profile(xi, sigma, gamma, dtype=np.float32)
    A0 = np.mean(A, axis=0, keepdims=True)
    A -= A0

    M = np.zeros(shape + (n_wv, n_wv))
    N = np.zeros(shape + (n_wv, n_wv))

    for k in range(n_wv_):
        gk = voigt_profile(wv_[k] - wvc - shift, sigma, gamma) ** 2

        for j in range(n_wv):
            mjk = gk * (voigt_profile(wv_[k] - wv[j] - shift, sigma, gamma) - A0[0,k])
            njk = gk * A[j,k]

            for i in range(n_wv):
                M[..., j, i] += mjk * A[i,k]
                N[..., j, i] += njk * A[i,k]

    Q = M @ np.linalg.inv(N + lam * np.identity(n_wv))
    return Q @ (np.identity(n_wv) - 1 / n_wv) + 1 / n_wv
