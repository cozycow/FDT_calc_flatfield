import numpy as np
from lmfit import lmfit
from scipy.special import wofz


def fit_prefilter(f1, f2, wv, Wmu=0.06, sigma0=0.043, gamma0=0.053, axis=-1, **kwargs):

    f1_ = np.moveaxis(f1.copy(), axis, -1)
    f2_ = np.moveaxis(f2.copy(), axis, -1)
    wv_ = np.moveaxis(wv.copy(), axis, -1)

    if len(f1_.shape) == 1:
        f1_ = np.expand_dims(f1_, 0)

    while len(f2_.shape) < len(f1_.shape):
        f2_ = np.expand_dims(f2_, 0)

    while len(wv_.shape) < len(f1_.shape):
        wv_ = np.expand_dims(wv_, 0)

    shift0 = np.nanmedian(np.take_along_axis(wv_, np.argmin(f1_, axis=-1, keepdims=True)))
    shift2 = np.nanmedian(np.take_along_axis(wv_, np.argmin(f2_, axis=-1, keepdims=True)))
    delta0 = np.nanmedian(shift2 - shift0)
    #print(delta0)

    #offset0 = np.nanmedian(np.max(f1_, axis=-1))

    local_params = np.moveaxis(np.array([np.ones_like(f1_[...,0]) * shift0,
                                         np.ones_like(f1_[...,0]) * sigma0,
                                         ]), 0, -1)

    global_params = np.array([delta0])
    local_params, global_params = lmfit(double_voigt, (wv_, f1_, f2_), local_params,
                                        global_params,
                                        **kwargs)

    return np.moveaxis(np.squeeze(local_params), -1, axis), global_params


def double_voigt(X, shift, sigma, delta, *args, **kwargs):
    wv, f1, f2 = X

    V1 = voigt_func((wv, 0), shift, sigma, delta, *args, **kwargs)
    V2 = voigt_func((wv, 0), shift + delta, sigma, delta, *args, **kwargs)
    V2[...,3] = V2[...,1]

    return V1 / np.expand_dims(f1, -1) - V2 / np.expand_dims(f2, -1)


def voigt_func(X, shift, sigma, *args, **kwargs):
    gamma = 0.053
    Wmu = 0.075

    wv, f = X
    wvc = wv - shift

    offset = 1
    height = -offset * Wmu

    z = (wvc + 1j * gamma) / sigma / np.sqrt(2)
    w = wofz(z)
    Rew, Imw = np.real(w), np.imag(w)

    V = Rew / np.sqrt(2 * np.pi) / sigma
    V_shift = (Rew * wvc - gamma * Imw) / np.sqrt(2 * np.pi) / sigma ** 3
    V_sigma = ((wvc ** 2 - gamma ** 2 - sigma ** 2) * Rew -
               2 * wvc * gamma * Imw +
               gamma * sigma * np.sqrt(2 / np.pi)) / np.sqrt(2 * np.pi) / sigma ** 4
    #V_gamma = -(sigma * np.sqrt(2 / np.pi) - wvc * Imw - gamma * Rew) / np.sqrt(2 * np.pi) / sigma ** 3

    return np.stack([V * height + offset - f,
                     V_shift * height,
                     V_sigma * height,
                     #-V * offset,
                     #V_gamma * height,
                     ] + [np.zeros_like(Rew) for arg in args],
                    axis=-1)

