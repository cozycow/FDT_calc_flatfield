import numpy as np
from lmfit import lmfit
from scipy.special import wofz


def fit_voigt(f, x, sigma0, height0=1., gamma0=0., axis=-1, **kwargs):
    f_ = np.moveaxis(f.copy(), axis, -1)
    x_ = np.moveaxis(x.copy(), axis, -1)

    if len(f.shape) == 1:
        f_ = np.expand_dims(f_, 0)

    while len(x_.shape) < len(f_.shape):
        x_ = np.expand_dims(x_, 0)

    shift0 = np.take_along_axis(x_,
                                np.argmax(f_, axis=-1, keepdims=True) * (height0 > 0) +
                                np.argmin(f_, axis=-1, keepdims=True) * (height0 < 0), axis=-1)[...,0]

    offset0 = np.min(f_, axis=-1) * (height0 > 0) + np.max(f_, axis=-1) * (height0 < 0)

    local_params = np.moveaxis(np.array([shift0,
                                         np.ones_like(shift0) * sigma0,
                                         offset0,
                                         np.ones_like(shift0) * height0,
                                         #np.ones_like(shift0) * gamma0
                                         ]), 0, -1)

    global_params = np.array([gamma0])
    local_params, global_params = lmfit(voigt_func, (x_, f_), local_params,
                                        global_params,
                                        **kwargs)

    return np.moveaxis(np.squeeze(local_params), -1, axis), global_params


def voigt_func(X, shift, sigma, offset, height, gamma, *args, **kwargs):
    wv, f = X

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

    return np.stack([V * height + offset - f,
                     V_shift * height,
                     V_sigma * height,
                     np.ones_like(V),
                     V,
                     V_gamma * height,
                     ] + [np.zeros_like(Rew) for arg in args],
                    axis=-1)

