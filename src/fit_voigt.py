import numpy as np
from lmfit import lmfit
from scipy.special import wofz


def fit_voigt(f, x, sigma0, height0=1., gamma0=0., axis=0, **kwargs):
    f_ = np.moveaxis(f.copy(), axis, 0)
    x_ = np.moveaxis(x.copy(), axis, 0)

    if len(f.shape) == 1:
        f_ = expand_dims(f_, axis=1)

    x_ = expand_dims(x_, f_, axis=1)

    shift0 = np.take_along_axis(x_,
                                np.argmax(f_, axis=0, keepdims=True) * (height0 > 0) +
                                np.argmin(f_, axis=0, keepdims=True) * (height0 < 0), axis=0)[0]

    offset0 = np.min(f_, axis=0) * (height0 > 0) + np.max(f_, axis=0) * (height0 < 0)

    local_params = np.array([shift0,
                             np.ones_like(shift0) * sigma0,
                             offset0,
                             np.ones_like(shift0) * height0,
                             #np.ones_like(shift0) * gamma0
                             ])

    global_params = expand_dims(np.array([gamma0]), local_params, 1)

    local_params, global_params, res = lmfit(voigt_func, x_, f_,
                                             local_params,
                                             global_params,
                                             **kwargs)

    return (np.moveaxis(local_params, 0, axis),
            global_params,
            np.nanstd(res, axis=0))


def voigt_func(wv, shift, sigma, offset, height, gamma, **kwargs):

    xc = wv - shift
    z = (xc + 1j * gamma) / sigma / np.sqrt(2)
    w = wofz(z)
    Rew, Imw = np.real(w), np.imag(w)

    V = Rew / np.sqrt(2 * np.pi) / sigma
    V_shift = (Rew * xc - gamma * Imw) / np.sqrt(2 * np.pi) / sigma ** 3
    V_sigma = ((xc ** 2 - gamma ** 2 - sigma ** 2) * Rew -
               2 * xc * gamma * Imw +
               gamma * sigma * np.sqrt(2 / np.pi)) / np.sqrt(2 * np.pi) / sigma ** 4
    V_gamma = -(sigma * np.sqrt(2 / np.pi) - xc * Imw - gamma * Rew) / np.sqrt(2 * np.pi) / sigma ** 3

    return (V * height + offset,
            np.array([V_shift * height,
                      V_sigma * height,
                      np.ones_like(V),
                      V,
                      V_gamma * height,
                      ]))


def expand_dims(a, b=None, axis=-1):
    if b is None:
        return np.expand_dims(a, axis=axis)
    elif len(a.shape) < len(b.shape):
        return expand_dims(np.expand_dims(a, axis=axis), b, axis=axis)
    else:
        return a

