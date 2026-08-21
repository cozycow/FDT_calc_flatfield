import numpy as np


def polyterms2d(x, y, degree=1):
    if degree == 0:
        return np.array([np.ones_like(x)])
    else:
        return np.append(polyterms2d(x, y, degree=degree-1), np.array([x ** (degree - i) * y ** i for i in range(degree + 1)]), axis=0)


def polyval2d(x, y, p):
    degree = int((np.sqrt(8 * len(p) + 1) - 3) / 2)
    return np.sum([p_ * x_ for p_, x_ in zip(p, polyterms2d(x, y, degree=degree))], axis=0)


def polyfit2d(f, x=None, y=None, degree=1, weight=None, return_coefficients=False):
    if x is None and y is None:
        nx, ny = f.shape
        x, y = np.mgrid[-nx // 2 + 0.5:nx // 2 + 0.5, -ny // 2 + 0.5:ny // 2 + 0.5].astype(np.float32)
        x /= nx / 2
        y /= ny / 2

    if weight is None:
        W = np.ones_like(f)
    else:
        W = weight

    t = np.where(~np.isnan(f))
    X = polyterms2d(x[t], y[t], degree=degree)[1:]
    Y = f[t]
    W = W[t]
    W /= np.sum(W)

    X0 = np.sum(X * W, axis=-1, keepdims=True)
    Y0 = np.sum(Y * W)

    X_ = X - X0
    Y_ = Y - Y0

    k = (Y_ * W) @ X_.T @ np.linalg.inv((X_ * W) @ X_.T)
    b = Y0 - k @ X0
    p = np.append(b, k)

    if return_coefficients:
        return p
    else:
        return polyval2d(x, y, p)


def interpolate(f, x, x_new):
    idx = np.searchsorted(x, x_new).clip(1, len(x) - 1)
    xa, xb = x[idx - 1], x[idx]
    dx = xb - xa

    a, b = (xb - x_new) / dx, (x_new - xa) / dx
    fa = np.take_along_axis(f, idx - 1, axis=0)
    fb = np.take_along_axis(f, idx, axis=0)
    return fa * a + fb * b