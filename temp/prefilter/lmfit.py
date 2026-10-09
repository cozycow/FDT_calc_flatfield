import numpy as np


def lmfit(func, x, local_inits, global_inits=None, *, niter, **kwargs):
    if global_inits is None:
        global_inits = np.array([0.])

    local_params = np.expand_dims(local_inits, -1)
    global_params = np.expand_dims(global_inits, -1)

    while len(global_params.shape) < len(local_params.shape):
        global_params = np.expand_dims(global_params, 0)

    nlocal = local_params.shape[-2]
    nglobal = global_params.shape[-2]

    for i in range(niter):
        res, J = func(x, *np.moveaxis(local_params, -2, 0),
                *np.moveaxis(global_params, -2, 0), **kwargs)

        delta_local, delta_global = solve(J[...,:nlocal], J[...,nlocal:nlocal+nglobal], -np.expand_dims(res, -1), **kwargs)

        local_params += delta_local
        global_params += delta_global

    return np.squeeze(local_params, -1), np.squeeze(global_params), res


def solve(Jl, Jg, y, *, lam, **kwargs):
    local_axes = tuple(range(len(y.shape) - 2))

    A = np.swapaxes(Jl, -1, -2) @ Jl
    A = np.linalg.inv(A + lam * np.identity(A.shape[-1]) * A + 1e-15 * np.identity(A.shape[-1]))
    u = np.swapaxes(Jl, -1, -2) @ y
    dl = A @ u

    W = 1 / (np.swapaxes(y, -1, -2) @ y + 1e-15)

    B = np.swapaxes(Jl, -1, -2) @ Jg
    BT = np.swapaxes(Jg, -1, -2) @ Jl
    C = A @ B

    D = np.nanmean((np.swapaxes(Jg , -1, -2) @ Jg - BT @ C) * W, axis=local_axes, keepdims=True)
    D = np.linalg.inv(D + 1e-15 * np.identity(D.shape[-1]))
    v = np.nanmean((np.swapaxes(Jg, -1, -2) @ y - BT @ dl) * W, axis=local_axes, keepdims=True)
    dg = D @ v
    dl -= C @ dg

    return dl, dg
