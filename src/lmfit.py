import numpy as np


def lstsq_loss(y, f, J):
    return f - y, J


def lmfit(func, x, y,
          local_inits,
          global_inits=np.array([]),
          loss=lstsq_loss,
          niter=1, **kwargs):

    local_params = np.expand_dims(local_inits, 1)
    global_params = np.expand_dims(global_inits, 1)

    nlocal = len(local_params)
    nglobal = len(global_params)

    for i in range(niter):
        f, J = func(x, *local_params, *global_params, **kwargs)
        res, J = loss(y, f, J)

        delta_local, delta_global = solve(np.moveaxis(J, (0,1), (-1,-2)),
                                          -np.expand_dims(np.moveaxis(res, 0, -1), -1),
                                          np.arange(nlocal, nlocal + nglobal),
                                          **kwargs)

        local_params += np.moveaxis(delta_local, (-2,-1), (0,1))
        global_params += np.moveaxis(delta_global, (-2,-1), (0,1))

    return np.squeeze(local_params, 1), np.squeeze(global_params, 1), res


def solve(J, y,
          global_idx=np.array([]), lam=0., **kwargs):

    n = J.shape[-1]
    local_idx = np.delete(np.arange(n), global_idx)
    local_axes = tuple(np.arange(len(y.shape) - 2))

    Jl = J[...,local_idx]
    Jg = J[...,global_idx]

    A = np.swapaxes(Jl, -1, -2) @ Jl
    A = np.linalg.inv(A + lam * np.identity(A.shape[-1]) * A + 1e-15 * np.identity(A.shape[-1]))
    u = np.swapaxes(Jl, -1, -2) @ y
    dl = A @ u

    B = np.swapaxes(Jl, -1, -2) @ Jg
    BT = np.swapaxes(B, -1, -2)
    C = A @ B

    W = 1 / (np.swapaxes(y, -1, -2) @ y + 1e-15)

    D = np.nanmean((np.swapaxes(Jg , -1, -2) @ Jg - BT @ C) * W, axis=local_axes, keepdims=True)
    D = np.linalg.inv(D + 1e-15 * np.identity(D.shape[-1]))
    v = np.nanmean((np.swapaxes(Jg, -1, -2) @ y - BT @ dl) * W, axis=local_axes, keepdims=True)
    dg = D @ v
    dl -= C @ dg

    return dl, dg

