import numpy as np


def fit_pv(f, x, fwhm0, eta, axis=-1, negative=False, **kwargs):
    f_ = np.moveaxis(f.copy(), axis, -1)
    x_ = np.moveaxis(x.copy(), axis, -1)

    if negative:
        f_ *= -1

    if len(f.shape) == 1:
        f_ = np.expand_dims(f_, 0)

    while len(x_.shape) < len(f_.shape):
        x_ = np.expand_dims(x_, 0)

    af, bf = np.min(f_, axis=-1), np.max(f_, axis=-1)
    ax, bx = np.min(x_, axis=-1), np.max(x_, axis=-1)

    af, bf = (bf + af) / 2, bf - af
    ax, bx = (bx + ax) / 2, bx - ax

    f_ = (f_ - np.expand_dims(af, -1)) / np.expand_dims(bf, -1)
    x_ = (x_ - np.expand_dims(ax, -1)) / np.expand_dims(bx, -1)

    shift0 = np.squeeze(np.take_along_axis(x_, np.argmax(f_, axis=-1, keepdims=True), axis=-1))
    p0 = np.moveaxis(np.array([shift0,
                               np.ones_like(shift0) * fwhm0 / bx,
                               np.ones_like(shift0) * fwhm0 / bx,
                               -np.ones_like(shift0) * 0.5,
                               #np.ones_like(shift0) * 0.5
                               ]), 0, -1)

    params = lmfit(pvfunc, pvjac, x_, f_, p0, eta=eta, **kwargs)

    params[...,0] = params[...,0] * bx + ax
    params[...,1] = params[...,1] * bx
    params[...,2] = params[...,2] * bf * bx
    params[...,3] = params[...,3] * bf + af

    if negative:
        params[...,2] *= -1
        params[...,3] *= -1

    return np.moveaxis(np.squeeze(params), -1, axis)


def lmfit(func, jac, x, y, p0, *, lam, niter, **kwargs):
    p = np.expand_dims(p0, -1)

    for i in range(niter):
        f, J = func(x, *np.moveaxis(p, -2, 0), **kwargs), jac(x, *np.moveaxis(p, -2, 0), **kwargs)
        A = np.moveaxis(J, -1, -2) @ J
        A += lam * np.identity(A.shape[-1]) * A
        b = np.moveaxis(J, -1, -2) @ np.expand_dims(y - f, -1)
        p += np.linalg.solve(A, b)

    return np.squeeze(p)


def pvfunc(wv, shift, fwhm, height, offset, *, eta, **kwargs):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    gamma = fwhm / 2

    L = gamma / np.pi / ((wv - shift) ** 2 + gamma ** 2)
    G = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(wv - shift) ** 2 / 2 / sigma ** 2)
    func = height * (eta * L + (1 - eta) * G) + offset
    return func


def pvjac(wv, shift, fwhm, height, offset, *, eta, **kwargs):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    gamma = fwhm / 2

    L = gamma / np.pi / ((wv - shift) ** 2 + gamma ** 2)
    G = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(wv - shift) ** 2 / 2 / sigma ** 2)

    L_shift = L ** 2 * (wv - shift) * 2 / gamma * np.pi
    G_shift = G * (wv - shift) / sigma ** 2

    L_fwhm = L * (1 / gamma - L * np.pi * 2) / 2
    G_fwhm = G * (-1 / sigma + (wv - shift) ** 2 / sigma ** 3) / 2 / np.sqrt(2 * np.log(2))

    jac = np.moveaxis(np.array([height * (eta * L_shift + (1 - eta) * G_shift),
                                height * (eta * L_fwhm + (1 - eta) * G_fwhm),
                                eta * L + (1 - eta) * G,
                                np.ones_like(L),
                                #height * (L - G)
                                ]), 0, -1)
    return jac

