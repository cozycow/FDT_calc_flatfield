import numpy as np


def read_wavelengths(header):
    nwv = header['WAVENUM']
    wv = []
    for i in range(nwv):
        wv.append(header[f'WAVELN{i + 1:02d}'])
    return np.array(wv)


def get_wavelengths(header, fg_data, update_header=False, **kwargs):
    fg_temp = header['FGOV1PT1']
    voltages = get_voltages(fg_data)
    wv = to_wavelength(voltages, fg_temp, **kwargs)
    if update_header:
        set_wavelegths(header, wv)
    return wv


def set_wavelegths(header, wv):
    if 'WAVENUM' in header:
        for i in range(header['WAVENUM']):
            del header['WAVELN' + ('%02d' % (i + 1))]

    header['WAVENUM'] = len(wv)
    for i in range(len(wv)):
        header['WAVELN' + ('%02d' % (i + 1))] = round(wv[i], 4)
    if len(wv) > 1:
        header['CONTPOS'] = int(np.where(np.argmax(np.abs(np.diff(wv))) != 0, len(wv), 1))
        if header['CONTPOS'] == 1:
            header['CONTPOSN'] = 'blue'
        else:
            header['CONTPOSN'] = 'red'
    else:
        header['CONTPOS'] = -1
        header['CONTPOSN'] = 'unknown'


def get_voltages(fg_data):
    voltages = np.sort(fg_data['PHI_FG_voltage'])
    t = np.where(voltages[1:] - voltages[:-1] > 60)[0] + 1

    x_ = 0
    out = []
    for x in t:
        out += [np.median(voltages[x_:x])]
        x_ = x
    out += [np.median(voltages[x_:])]

    return np.array(out)


def to_wavelength(voltage, temperature,
                  temperature_constant = 4.01225e-2,
                  tuning_constant = 3.513e-4,
                  ref_wavelength = 6173.341,
                  T0 = 61,
                  **kwargs):

    return ref_wavelength + tuning_constant * voltage + temperature_constant * (temperature - T0)
