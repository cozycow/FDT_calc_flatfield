import numpy as np
from wavelengths import read_wavelengths
from fit_voigt import *


def classical_estimates(data, header, **kwargs):
    q_V = 299792458 / 6173.341
    q_B = q_V * 0.231

    data_ = data.copy().reshape(-1, 4, data.shape[-2], data.shape[-1])

    lcp = (data_[:,0] + data_[:,3]) / 2
    rcp = (data_[:,0] - data_[:,3]) / 2

    v_lcp = get_wv_shift(lcp, header, **kwargs)
    v_rcp = get_wv_shift(rcp, header, **kwargs)

    return q_B * (v_lcp - v_rcp), q_V * (v_lcp + v_rcp) / 2


def get_wv_shift(data, header, **kwargs):
    line_params = fit_line(data, header, correct_doppler=True, **kwargs)
    return line_params[0].clip(-0.5,0.5)


def fit_line(data, header, pol=0, lam=1e-3, niter=10, sigma0=0.043, gamma0=0.053, Wmu=0.06, **kwargs):
    contpos = header['CONTPOS'] - 1

    wvlns = read_wavelengths(header, **kwargs)
    wvlns -= 6173.341  # header['WAVELNTH']

    nwv = len(wvlns)
    nx, ny = data.shape[-2:]
    data_ = data.copy().reshape(nwv, -1, nx, ny)[:, pol]
    height0 = -np.nanmedian(data_[contpos]) * Wmu

    return fit_voigt(data_, wvlns,
                     #np.delete(data_, contpos, axis=0), np.delete(wvlns, contpos, axis=0),
                     sigma0, height0=height0, gamma0=gamma0,
                     #weight=np.expand_dims(data_[contpos].clip(0), (-1,-2)),
                     axis=0, lam=lam, niter=niter, **kwargs)


