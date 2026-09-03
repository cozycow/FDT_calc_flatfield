import numpy as np
from fit_pv import *


def classical_estimates(data, header, **kwargs):
    q_V = 299792458 / 6173.341
    q_B = q_V * 0.231

    data_ = data.copy().reshape(-1, 4, data.shape[-2], data.shape[-1])

    lcp = (data_[:,0] + data_[:,3]) / 2
    rcp = (data_[:,0] - data_[:,3]) / 2

    v_lcp = get_wv_shift(lcp, header, **kwargs)
    v_rcp = get_wv_shift(rcp, header, **kwargs)

    return q_B * (v_lcp - v_rcp), q_V * (v_lcp + v_rcp) / 2


def get_wv_shift(data, header, pol=0, **kwargs):

    if 'wavelengths' in kwargs:
        wavelengths = kwargs['wavelengths']
    elif 'WAVENUM' in header:
        nwv = header['WAVENUM']
        wavelengths = []
        for i in range(nwv):
            wavelengths.append(header[f'WAVELN{i + 1:02d}'])
        wavelengths = np.array(wavelengths)
    else:
        raise ValueError('Wavelengths not provided')

    if 'contpos' in kwargs:
        contpos = kwargs['contpos']
    elif 'CONTPOS' in header:
        contpos = header['CONTPOS'] - 1
    else:
        raise ValueError('Continuum position not provided')

    wv0 = np.mean(np.delete(wavelengths, contpos))

    nx, ny = data.shape[-2:]
    data_ = data.copy().reshape(6, -1, nx, ny)[:, pol]
    line_params = fit_pv(data_, wavelengths - wv0, axis=0, negative=True, **kwargs)
    return line_params[0]
