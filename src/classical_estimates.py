import numpy as np


def classical_estimates(data, header):
    q_V = 299792458 / 6173.3433
    q_B = q_V * 0.231

    data_ = data.copy().reshape(-1, 4, data.shape[-2], data.shape[-1])

    lcp = (data_[:,0] + data_[:,3]) / 2
    rcp = (data_[:,0] - data_[:,3]) / 2

    v_lcp = get_wv_shift(lcp, header)
    v_rcp = get_wv_shift(rcp, header)

    return q_B * (v_lcp - v_rcp), q_V * (v_lcp + v_rcp) / 2


def get_wv_shift(data, header, pol=0, log=True, **kwargs):

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

    wavelengths = np.delete(wavelengths, contpos)
    delta_wv = np.mean(wavelengths[1:] - wavelengths[:-1])

    temp = data.copy().reshape((6, -1, data.shape[-2], data.shape[-1]))[:, pol]

    if log:
        temp = -np.log((temp[contpos] - np.delete(temp, contpos, axis=0)).clip(1))
    else:
        temp = np.delete(temp, contpos, axis=0)

    t = np.argmin(temp, axis=0)
    l, a, r = np.take_along_axis(temp, np.array([(t - 1) % 5, t, (t + 1) % 5]), axis=0)
    b, c = (r - l) / 2, (l + r) - 2 * a

    with np.errstate(invalid='ignore'):
        return np.nan_to_num(t - 2 - b / c) * delta_wv
