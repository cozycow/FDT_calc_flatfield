import numpy as np


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
                                         #np.ones_like(shift0) * eta0
                                         ]), 0, -1)

    global_params = np.array([eta0])
    while len(global_params.shape) < len(local_params.shape):
        global_params = np.expand_dims(global_params, 0)

    local_params, global_params = lmfit(pvfunc, pvjac, x_, f_, local_params, global_params, **kwargs)

    return np.moveaxis(np.squeeze(local_params), -1, axis), global_params


def lmfit(func, jac, x, y, local_inits, global_inits, *, niter, **kwargs):
    local_params = np.expand_dims(local_inits, -1)
    global_params = np.expand_dims(global_inits, -1)
    nlocal = local_params.shape[-2]
    nglobal = global_params.shape[-2]

    for i in range(niter):
        f = func(x, *np.moveaxis(local_params, -2, 0),
                 *np.moveaxis(global_params, -2, 0), **kwargs)
        J = jac(x, *np.moveaxis(local_params, -2, 0),
                *np.moveaxis(global_params, -2, 0), **kwargs)

        delta_local, delta_global = solve(J[...,:nlocal], J[...,nlocal:nlocal+nglobal], np.expand_dims(y - f, -1), **kwargs)

        local_params += delta_local
        global_params += delta_global

    return np.squeeze(local_params), np.squeeze(global_params)


def solve(Jl, Jg, y, *, lam, **kwargs):
    local_axes = tuple(range(len(y.shape) - 2))

    A = np.swapaxes(Jl, -1, -2) @ Jl
    A = np.linalg.inv(A + lam * np.identity(A.shape[-1]) * A + 1e-16 * np.identity(A.shape[-1]))
    u = np.swapaxes(Jl, -1, -2) @ y
    dl = A @ u

    B = np.swapaxes(Jl, -1, -2) @ Jg
    BT = np.swapaxes(B, -1, -2)
    C = A @ B

    D = np.nanmean(np.swapaxes(Jg, -1, -2) @ Jg - BT @ C, axis=local_axes, keepdims=True)
    D = np.linalg.inv(D + 1e-16 * np.identity(D.shape[-1]))
    v = np.nanmean(np.swapaxes(Jg, -1, -2) @ y - BT @ dl, axis=local_axes, keepdims=True)
    dg = D @ v
    dl -= C @ dg

    return dl, dg


def pvfunc(wv, shift, fwhm, offset, height, eta, *args, **kwargs):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    gamma = fwhm / 2

    L = gamma / np.pi / ((wv - shift) ** 2 + gamma ** 2)
    G = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(wv - shift) ** 2 / 2 / sigma ** 2)
    func = height * (eta * L + (1 - eta) * G) + offset
    return func


def pvjac(wv, shift, fwhm, offset, height, eta, *args, **kwargs):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    gamma = fwhm / 2

    L = gamma / np.pi / ((wv - shift) ** 2 + gamma ** 2)
    G = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(wv - shift) ** 2 / 2 / sigma ** 2)

    L_shift = L ** 2 * (wv - shift) * 2 / gamma * np.pi
    G_shift = G * (wv - shift) / sigma ** 2

    L_fwhm = L * (1 / gamma - L * np.pi * 2) / 2
    G_fwhm = G * (-1 / sigma + (wv - shift) ** 2 / sigma ** 3) / 2 / np.sqrt(2 * np.log(2))

    return np.stack([height * (eta * L_shift + (1 - eta) * G_shift),
                     height * (eta * L_fwhm + (1 - eta) * G_fwhm),
                     np.ones_like(L),
                     eta * L + (1 - eta) * G,
                     height * (L - G)] +
                    [np.zeros_like(L) for arg in args],
                    axis=-1)

