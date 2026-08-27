import numpy as np


def classical_estimates(data, header):
    q_V = 299792458 / 6173.341
    q_B = q_V * 0.231

    data_ = data.copy().reshape(-1, 4, data.shape[-2], data.shape[-1])

    lcp = (data_[:,0] + data_[:,3]) / 2
    rcp = (data_[:,0] - data_[:,3]) / 2

    v_lcp = get_wv_shift(lcp, header)
    v_rcp = get_wv_shift(rcp, header)

    return q_B * (v_lcp - v_rcp), q_V * (v_lcp + v_rcp) / 2


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

    continuum = calc_continuum(data, header)

    wavelengths = np.delete(wavelengths, contpos)
    delta_wv = np.mean(wavelengths[1:] - wavelengths[:-1])

    temp = data.copy().reshape((6, -1, data.shape[-2], data.shape[-1]))[:, pol]

    if log:
        temp = -np.log(np.abs(continuum - np.delete(temp, contpos, axis=0)))
    else:
        temp = np.delete(temp, contpos, axis=0)

    t = np.argmin(temp, axis=0)
    l, a, r = np.take_along_axis(temp, np.array([(t - 1) % 5, t, (t + 1) % 5]), axis=0)
    b, c = (r - l) / 2, (l + r) - 2 * a

    with np.errstate(invalid='ignore'):
        return np.nan_to_num(t - 2 - b / c) * delta_wv


def calc_continuum(data, header, n_comp=101, sigma=0.043, gamma=0.053, lam=1e-6, **kwargs):
    from scipy.special import voigt_profile

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

    n = len(wavelengths)
    x_min = np.min(np.delete(wavelengths, contpos))
    x_max = np.max(np.delete(wavelengths, contpos))
    dx = (x_max - x_min) / (n_comp - 1)
    xc = (x_min + x_max) / 2
    x = np.arange(x_min, x_max + dx / 2, dx, dtype=np.float32)

    A = voigt_profile(np.expand_dims(wavelengths, axis=1) - np.expand_dims(x, axis=0), sigma, gamma, dtype=np.float32)
    A0 = np.mean(A, axis=0, keepdims=True)
    A = A - A0

    W = np.diag(voigt_profile(x - xc, sigma, gamma) ** 2)
    q = 1 / n - A0 @ W @ A.T @ np.linalg.inv(A @ W @ A.T + lam * np.identity(n)) @ (np.identity(n) - 1 / n)


    nx, ny = data.shape[-2:]
    return np.linalg.tensordot(data.reshape(n,-1,nx,ny), q[0], axes=(0, 0))
