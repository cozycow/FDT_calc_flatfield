import numpy as np
from lmfit import lmfit
from scipy.special import wofz


def fit_prefilter(f, wv, mu, Wmu=0.06, sigma0=0.043, gamma0=0.053, axis=-1, **kwargs):

    f_ = np.moveaxis(f.copy(), axis, -1)
    wv_ = np.moveaxis(wv.copy(), axis, -1)
    mu_ = np.moveaxis(mu.copy(), axis, -1)

    if len(f.shape) == 1:
        f_ = np.expand_dims(f_, 0)

    while len(wv_.shape) < len(f_.shape):
        wv_ = np.expand_dims(wv_, 0)

    while len(mu_.shape) < len(f_.shape):
        mu_ = np.expand_dims(mu_, 0)

    shift0 = np.nanmedian(np.take_along_axis(wv_, np.argmin(f_, axis=-1, keepdims=True)))
    offset0 = np.nanmedian(np.max(f_, axis=-1))

    local_params = np.moveaxis(np.array([np.ones_like(f_[...,0]) * shift0,
                                         np.ones_like(f_[...,0]) * offset0,
                                         ]), 0, -1)

    global_params = np.array([Wmu, gamma0, sigma0, 0.])
    local_params, global_params = lmfit(voigt_func, (wv_, mu_, f_), local_params,
                                        global_params,
                                        **kwargs)

    return np.moveaxis(np.squeeze(local_params), -1, axis), global_params


def voigt_func(X, shift, offset, Wmu, gamma, sigma0, sigma_mu, *args, **kwargs):
    wv, mu, f = X
    wvc = wv - shift
    height = -offset * Wmu
    sigma = sigma0 + sigma_mu * mu

    z = (wvc + 1j * gamma) / sigma / np.sqrt(2)
    w = wofz(z)
    Rew, Imw = np.real(w), np.imag(w)

    V = Rew / np.sqrt(2 * np.pi) / sigma
    V_shift = (Rew * wvc - gamma * Imw) / np.sqrt(2 * np.pi) / sigma ** 3
    V_sigma = ((wvc ** 2 - gamma ** 2 - sigma ** 2) * Rew -
               2 * wvc * gamma * Imw +
               gamma * sigma * np.sqrt(2 / np.pi)) / np.sqrt(2 * np.pi) / sigma ** 4
    V_gamma = -(sigma * np.sqrt(2 / np.pi) - wvc * Imw - gamma * Rew) / np.sqrt(2 * np.pi) / sigma ** 3

    return np.stack([V * height + offset - f,
                     V_shift * height,
                     np.ones_like(V),
                     -V * offset,
                     V_gamma * height,
                     V_sigma * height,
                     V_sigma * height * mu,
                     ] + [np.zeros_like(Rew) for arg in args],
                    axis=-1)

