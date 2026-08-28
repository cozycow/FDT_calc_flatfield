import numpy as np


def fit_pv(f, x, **kwargs):
    n = len(f)

    shift0 = find_shift_coarse(f, x)
    fwhm0 = (np.max(x, axis=-1) - np.min(x, axis=-1)) / 2 * np.ones(n)
    depth0 = (np.min(f, axis=-1) - np.max(f, axis=-1)) * fwhm0
    eta0 = 0.5 * np.ones(n)
    offset0 = np.max(f, axis=-1)

    p0 = np.array([shift0, fwhm0, depth0, eta0, offset0]).T
    params = lmfit(pseudoVoigt, x, f, p0, **kwargs)
    return params


def pseudoVoigt(wv, shift, fwhm, depth, eta, offset):
    sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
    gamma = fwhm / 2

    L = gamma / np.pi / ((wv - shift) ** 2 + gamma ** 2)
    G = 1 / sigma / np.sqrt(2 * np.pi) * np.exp(-(wv - shift) ** 2 / 2 / sigma ** 2)

    L_shift = L ** 2 * (wv - shift) * 2 / gamma * np.pi
    G_shift = G * (wv - shift) / sigma ** 2

    L_fwhm = L * (1 / gamma - L * np.pi * 2) / 2
    G_fwhm = G * (-1 / sigma + (wv - shift) ** 2 / sigma ** 3) / 2 / np.sqrt(2 * np.log(2))

    func = depth * (eta * L + (1 - eta) * G) + offset
    jac = np.array([depth * (eta * L_shift + (1 - eta) * G_shift),
                    depth * (eta * L_fwhm + (1 - eta) * G_fwhm),
                    eta * L + (1 - eta) * G,
                    depth * (L - G),
                    np.ones_like(L)])

    return func, jac


def lmfit(func_jac, x, y, p0, lam=1e-2, niter=5, **kwargs):
    p = np.expand_dims(p0, -1)

    for i in range(niter):
        f, j = func_jac(x, *np.transpose(p, (1, 0, 2)))

        A = np.matmul(j, j, axes=[(0,-1),(-1,0),(-2,-1)])
        A += lam * np.identity(p.shape[1]) * np.expand_dims(np.trace(A, axis1=-2, axis2=-1), (1, 2))
        b = np.sum(j * (y - f), axis=-1).T
        b = np.expand_dims(b, -1)
        p += np.linalg.solve(A, b)

    return p[...,0]


def find_shift_coarse(f, x):
    n = x.shape[-1]
    dx = np.median(x[...,1:] - x[...,:-1], axis=-1)

    t = np.argmin(f, axis=-1)
    l, a, r = np.take_along_axis(f.T, np.array([(t - 1) % n, t, (t + 1) % n]), axis=0)
    b, c = (r - l) / 2, (l + r) - 2 * a
    return (t - b / c) * dx + x[0]
