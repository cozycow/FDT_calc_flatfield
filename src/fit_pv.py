import numpy as np


def fit_pv(f, x, axis=-1, negative=False, **kwargs):
    f_ = np.moveaxis(f.copy(), axis, -1)
    x_ = np.moveaxis(x.copy(), axis, -1)

    if negative:
        f_ *= -1

    af, bf = np.min(f_, axis=-1, keepdims=True), np.max(f_, axis=-1, keepdims=True)
    ax, bx = np.min(x_, axis=-1, keepdims=True), np.max(x_, axis=-1, keepdims=True)
    af, bf = af, bf - af
    ax, bx = (bx + ax) / 2, bx - ax

    f_ = (f_ - af) / bf
    x_ = (x_ - ax) / bx

    if len(f.shape) == 1:
        f_ = np.expand_dims(f_, 0)

    while len(x_.shape) < len(f_.shape):
        x_ = np.expand_dims(x_, 0)

    shift0 = np.mean(f_ * x_, axis=-1) / np.mean(f_, axis=-1)
    fwhm0 = (np.sqrt(np.mean(f_ * (x_ - np.expand_dims(shift0, -1)) ** 2, axis=-1) / np.mean(f_, axis=-1)) *
             2 * np.sqrt(2 * np.log(2)))

    p0 = np.moveaxis(np.array([shift0,
                               np.ones_like(f_[...,0]) * fwhm0,
                               np.ones_like(f_[...,0]) * fwhm0,
                               np.zeros_like(f_[...,0]),
                               np.ones_like(f_[...,0]) * 0.5]), 0, -1)

    params = lmfit(pseudoVoigt, x_, f_, p0, **kwargs)
    params[...,0] = params[...,0] * np.squeeze(bx) + np.squeeze(ax)
    params[...,1] = params[...,1] * np.squeeze(bx)
    params[...,2] = params[...,2] * np.squeeze(bf) * np.squeeze(bx)
    params[...,3] = params[...,3] * np.squeeze(bf) + np.squeeze(af)

    if negative:
        params[...,2] *= -1
        params[...,3] *= -1

    return np.moveaxis(np.squeeze(params), -1, axis)


def lmfit(func_jac, x, y, p0, lam=1e-1, niter=10, **kwargs):
    p = np.expand_dims(p0, -1)

    for i in range(niter):
        f, j = func_jac(x, *np.moveaxis(p, -2, 0))
        A = np.moveaxis(j, -1, -2) @ j
        A += lam * np.identity(A.shape[-1]) * A
        b = np.moveaxis(j, -1, -2) @ np.expand_dims(y - f, -1)
        p += np.linalg.solve(A, b)

    return p[...,0]


def pseudoVoigt(wv, shift, fwhm, height, offset, eta):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    gamma = fwhm / 2

    L = gamma / np.pi / ((wv - shift) ** 2 + gamma ** 2)
    G = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(wv - shift) ** 2 / 2 / sigma ** 2)

    L_shift = L ** 2 * (wv - shift) * 2 / gamma * np.pi
    G_shift = G * (wv - shift) / sigma ** 2

    L_fwhm = L * (1 / gamma - L * np.pi * 2) / 2
    G_fwhm = G * (-1 / sigma + (wv - shift) ** 2 / sigma ** 3) / 2 / np.sqrt(2 * np.log(2))

    func = height * (eta * L + (1 - eta) * G) + offset
    jac = np.moveaxis(np.array([height * (eta * L_shift + (1 - eta) * G_shift),
                                height * (eta * L_fwhm + (1 - eta) * G_fwhm),
                                eta * L + (1 - eta) * G,
                                np.ones_like(L),
                                height * (L - G)]), 0, -1)

    return func, jac

