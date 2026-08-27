import numpy as np


def classical_estimates(data, header, **kwargs):
    q_V = 299792458 / 6173.341
    q_B = q_V * 0.231

    data_ = data.copy().reshape(-1, 4, data.shape[-2], data.shape[-1])

    lcp = (data_[:,0] + data_[:,3]) / 2
    rcp = (data_[:,0] - data_[:,3]) / 2

    v_lcp = get_wv_shift(lcp, header, **kwargs)
    v_rcp = get_wv_shift(rcp, header, **kwargs)

    return q_B * (v_lcp - v_rcp), q_V * (v_lcp + v_rcp) / 2


def get_wv_shift(data, header, pol=0, split=4, **kwargs):

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

    nx, ny = data.shape[-2:]
    mx, my = nx // split, ny // split
    data_ = data.copy().reshape((6, -1, nx, ny))[:, pol]
    data_ /= data_[contpos]

    shift = np.zeros((nx, ny))
    for i in range(split):
        for j in range(split):
            temp = data_[:, i * mx: (i + 1) * mx, j * my: (j + 1) * my]
            temp_, wv_ = deconvolve(temp, wavelengths, continuum=contpos, **kwargs)
            wv_ -= np.mean(wv_)
            with np.errstate(invalid='ignore'):
                shift[i * mx: (i + 1) * mx, j * my: (j + 1) * my] = np.mean(temp_[:-1] * np.expand_dims(wv_, axis=(1,2)), axis=0) / np.mean(temp_[:-1], axis=0)

    return shift


def deconvolve(f, x, x_out=None, continuum=-1, sigma=0.043, gamma=0.053,
               n_out=21, niter=3, lam=1e-10, p=2, **kwargs):
    from scipy.special import voigt_profile

    n_in = len(x)

    if x_out is None:
        x_min = np.min(np.delete(x, continuum) if continuum is not None else x)
        x_max = np.max(np.delete(x, continuum) if continuum is not None else x)
        dx = (x_max - x_min) / (n_out - 1)
        x_out = np.arange(x_min, x_max + dx / 2, dx)#, dtype=np.float32)
    else:
        n_out = len(x_out)

    xc = np.mean(x_out)

    A = voigt_profile(np.expand_dims(x, axis=1) - np.expand_dims(x_out, axis=0), sigma, gamma, dtype=np.float32)
    A0 = np.mean(A, axis=0, keepdims=True)
    A -= A0

    Q = np.zeros((n_in, n_in, n_out))#, dtype=np.float32)
    for i in range(n_in):
        for j in range(n_in):
            Q[i, j] = A[i] * A[j]

    f_ = np.moveaxis(f - np.mean(f, axis=0, keepdims=True), 0, -1)
    w = np.ones(f_.shape[:-1] + (n_out,), dtype=np.float32) * voigt_profile(x_out - xc, sigma, gamma)

    for i in range(niter):
        w = np.abs(w) ** p
        P = np.linalg.solve(np.tensordot(w, Q, axes=(-1, -1)) + np.identity(n_in) * lam,
                            np.expand_dims(f_, axis=-1))[...,0]
        w = w * np.tensordot(P, A, axes=(-1,-2))

    w0 = np.expand_dims(np.mean(f, axis=0) - np.tensordot(w, A0[0], axes=(-1, -1)), axis=-1)
    w = np.append(w, w0, axis=-1)
    w = np.moveaxis(w, -1, 0)

    return w, x_out
