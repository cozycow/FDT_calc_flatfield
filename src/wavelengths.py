import numpy as np
from datetime import datetime


def read_wavelengths(header, correct_doppler=False):
    nwv = header['WAVENUM']
    wvlns = []
    for i in range(nwv):
        wvlns.append(header[f'WAVELN{i + 1:02d}'])
    wvlns = np.array(wvlns)

    if correct_doppler:
        wvlns -= header['OBS_VR'] * 6173.341 / 299792458

    return wvlns


def get_wavelengths(header, fg_data, pmp_data, update_header=False, **kwargs):
    fg_temp = header['FGOV1PT1']
    voltages = get_mean_voltages(header, fg_data, pmp_data)
    wv = to_wavelength(voltages, fg_temp, **kwargs)
    if update_header:
        set_wavelegths(header, np.mean(wv, axis=0))
    return wv


def take_left(f, x, x_new):
    idx = np.searchsorted(x, x_new, side='right').clip(1, len(x))
    return np.take_along_axis(f, idx - 1, axis=0)


def calc_mean_voltages(fg_voltages, fg_times, pmp_times, acc_scheme):
    fg_voltages_ = take_left(fg_voltages, fg_times, pmp_times)
    return np.mean(fg_voltages_.reshape(acc_scheme), axis=(0,-1)).T


def get_mean_voltages(header, fg_data, pmp_data):
    fg_times = np.array([datetime.fromisoformat(temp) for temp in fg_data['RecordTime']])
    pmp_times = np.array([datetime.fromisoformat(temp) for temp in pmp_data['RecordTime']])
    fg_voltages = fg_data['PHI_FG_voltage'].astype(float)
    acc_scheme = (header['ACCROWIT'], header['ACCNROWS'], header['ACCNCOLS'], header['ACCCOLIT'])
    return calc_mean_voltages(fg_voltages, fg_times, pmp_times, acc_scheme)


def to_wavelength(x, temperature,
                  temperature_constant = 4.01225e-2,
                  tuning_constant = 3.513e-4,
                  ref_wavelength = 6173.341,
                  T0 = 61,
                  inv=False,
                  **kwargs):

    if inv:
        return (x - ref_wavelength - temperature_constant * (temperature - T0)) / tuning_constant
    else:
        return ref_wavelength + tuning_constant * x + temperature_constant * (temperature - T0)


def to_voltage(wavelength, temperature,
               **kwargs):
    return to_wavelength(wavelength, temperature, inv=True, **kwargs)


def set_wavelegths(header, wv):
    header['WAVENUM'] = len(wv)
    for i in range(len(wv)):
        header['WAVELN' + ('%02d' % (i + 1))] = round(wv[i], 5)
    if len(wv) > 1:
        header['CONTPOS'] = int(np.where(np.argmax(np.abs(np.diff(wv))) != 0, len(wv), 1))
        if header['CONTPOS'] == 1:
            header['CONTPOSN'] = 'blue'
        else:
            header['CONTPOSN'] = 'red'
    else:
        header['CONTPOS'] = -1
        header['CONTPOSN'] = 'unknown'

    ## need to update wavemin, wavemax, tuncons, tempcons
