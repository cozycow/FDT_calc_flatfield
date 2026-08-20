import numpy as np


def correct_crosstalk(data, header, **kwargs):
    continuum = calc_continuum(data, header, **kwargs)
    data_ = data.copy().reshape(-1,4,data.shape[-2],data.shape[-1])

    for i in range(data_.shape[0]):
        for j in range(1,4):
            data_[i,j] -= continuum[j] / continuum[0].clip(1) * data[i,0]
    return data_.reshape(data.shape)


def calc_continuum(data, header, n_comp=101, sigma=0.043, gamma=0.053, lam=1e-6, shift=0, return_coeff=False, **kwargs):
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
    x_min = np.min(np.delete(wavelengths, contpos)) + shift
    x_max = np.max(np.delete(wavelengths, contpos)) + shift
    dx = (x_max - x_min) / (n_comp - 1)
    xc = (x_min + x_max) / 2
    x = np.arange(x_min, x_max + dx / 2, dx, dtype=np.float32)

    A = voigt_profile(np.expand_dims(wavelengths, axis=1) - np.expand_dims(x, axis=0), sigma, gamma, dtype=np.float32)
    A0 = np.mean(A, axis=0, keepdims=True)
    A = A - A0

    W = np.diag(voigt_profile(x - xc, sigma, gamma) ** 2)
    q = 1 / n - A0 @ W @ A.T @ np.linalg.inv(A @ W @ A.T + lam * np.identity(n)) @ (np.identity(n) - 1 / n)

    if return_coeff:
        return q[0]
    else:
        nx, ny = data.shape[-2:]
        return np.linalg.tensordot(data.reshape(n,-1,nx,ny), q[0], axes=(0, 0))