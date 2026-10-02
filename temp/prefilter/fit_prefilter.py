import numpy as np
from lmfit import lmfit
from scipy.special import wofz


def fit_prefilter(f1, f2, f3, wv, mu, height0=-0.06, sigma0=0.043, gamma0=0.053, axis=-1, **kwargs):

    f1_ = np.moveaxis(f1.copy(), axis, -1)
    f2_ = np.moveaxis(f2.copy(), axis, -1)
    f3_ = np.moveaxis(f3.copy(), axis, -1)
    wv_ = np.moveaxis(wv.copy(), axis, -1)
    mu_ = np.moveaxis(mu.copy(), axis, -1)

    if len(f1_.shape) == 1:
        f1_ = np.expand_dims(f1_, 0)

    while len(f2_.shape) < len(f1_.shape):
        f2_ = np.expand_dims(f2_, 0)

    while len(f3_.shape) < len(f1_.shape):
        f3_ = np.expand_dims(f3_, 0)

    while len(wv_.shape) < len(f1_.shape):
        wv_ = np.expand_dims(wv_, 0)

    while len(mu_.shape) < len(f1_.shape):
        mu_ = np.expand_dims(mu_, 0)

    shift1 = np.nanmedian(np.take_along_axis(wv_, np.argmin(f1_, axis=-1, keepdims=True)))
    shift2 = np.nanmedian(np.take_along_axis(wv_, np.argmin(f2_, axis=-1, keepdims=True)))
    shift3 = np.nanmedian(np.take_along_axis(wv_, np.argmin(f3_, axis=-1, keepdims=True)))
    delta1 = np.nanmedian(shift2 - shift1)
    delta2 = np.nanmedian(shift3 - shift1)
    #print(delta0)

    #offset0 = np.nanmedian(np.max(f1_, axis=-1))

    local_params = np.moveaxis(np.array([np.ones_like(f1_[...,0]) * shift1,
                                         np.ones_like(f1_[..., 0]) * height0,
                                         #np.ones_like(f1_[...,0]) * sigma0,
                                         ]), 0, -1)

    global_params = np.array([sigma0, 0, gamma0, delta1, delta2])
    local_params, global_params, res = lmfit(triple_voigt, (wv_, mu_, f1_, f2_, f3_), local_params,
                                        global_params,
                                        **kwargs)

    return np.moveaxis(local_params, -1, axis), global_params, np.nanstd(res, axis=-1)


def triple_voigt(X, shift, height, sigma, sigma_mu, gamma, delta1, delta2, *args, **kwargs):
    wv, mu, f1, f2, f3 = X

    y1, V1 = voigt_func((wv, mu, 0), shift, height, sigma, sigma_mu, gamma, delta1, delta2, *args, **kwargs)
    y2, V2 = voigt_func((wv, mu, 0), shift + delta1, height, sigma, sigma_mu, gamma, delta1, delta2, *args, **kwargs)
    y3, V3 = voigt_func((wv, mu, 0), shift + delta2, height, sigma, sigma_mu, gamma, delta1, delta2, *args, **kwargs)

    V2[...,5] = V2[...,0]
    V3[...,6] = V3[...,0]

    return (y1 * (f2 + f3) - f1 * (y2 + y3),
            V1 * np.expand_dims(f2 + f3, -1) - (V2 + V3) * np.expand_dims(f1, -1))


def voigt_func(X, shift, height, sigma0, sigma_mu, gamma, *args, **kwargs):
    wv, mu, f = X
    wvc = wv - shift
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

    return (V * height + 1 - f,
            np.stack([V_shift * height,
                      V,
                      V_sigma * height,
                      V_sigma * height * mu,
                      V_gamma * height,
                      ] + [np.zeros_like(Rew) for arg in args],
                     axis=-1))

