import numpy as np
from lmfit import lmfit


def fit_pv(f, x, fwhm0, height0=1, eta0=0.5, axis=-1, **kwargs):
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
                                         np.ones_like(shift0) * fwhm0,
                                         offset0,
                                         np.ones_like(shift0) * height0,
                                         np.ones_like(shift0) * eta0
                                         ]), 0, -1)

    global_params = np.array([eta0])
    local_params, global_params = lmfit(pvfunc, x_, f_, local_params,
                                        #global_params,
                                        **kwargs)

    return np.moveaxis(np.squeeze(local_params), -1, axis), global_params


def pvfunc(wv, shift, fwhm, offset, height, eta, *args, **kwargs):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    gamma = fwhm / 2

    L = gamma / np.pi / ((wv - shift) ** 2 + gamma ** 2)
    G = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(wv - shift) ** 2 / 2 / sigma ** 2)
    V = eta * L + (1 - eta) * G

    L_shift = L ** 2 * (wv - shift) * 2 / gamma * np.pi
    G_shift = G * (wv - shift) / sigma ** 2

    L_fwhm = L * (1 / gamma - L * np.pi * 2) / 2
    G_fwhm = G * (-1 / sigma + (wv - shift) ** 2 / sigma ** 3) / 2 / np.sqrt(2 * np.log(2))

    return np.stack([height * V + offset,
                     height * (eta * L_shift + (1 - eta) * G_shift),
                     height * (eta * L_fwhm + (1 - eta) * G_fwhm),
                     np.ones_like(L),
                     V,
                     height * (L - G)] +
                    [np.zeros_like(L) for arg in args],
                    axis=-1)

