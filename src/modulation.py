import numpy as np


def modulate(data, header, inv=False, rotate=False):
    detector = header['DETECTOR']
    temperature = int(header['FPMPTSP1'])

    if rotate:
        angle = -127.6
    else:
        angle = 0

    O = modulation_matrix(detector, temperature, angle)

    if inv:
        O = np.linalg.inv(O)

    nx, ny = data.shape[-2:]
    data_ = data.copy().reshape((-1,4,nx,ny))
    data_ = np.matmul(O, data_, axes=[(-2, -1), (1, 0), (1, 0)])
    return data_.reshape(data.shape)


def demodulate(data, header, rotate=False):
    return modulate(data, header, inv=True, rotate=rotate)


def modulation_matrix(detector='FDT', temperature=45, angle=0):

    if detector == 'FDT':
        if temperature == 45:
            O = np.array([[1.0023, -0.64814, -0.56202, -0.51859],
                            [1.0041, 0.54693, -0.55299, 0.633],
                            [0.99523, 0.46132, 0.54165, -0.69603],
                            [0.99838, -0.61944, 0.66189, 0.42519]])
        elif temperature == 40:
            O = np.array([[0.99913, -0.69504, -0.38074, -0.60761],
                             [1.0051, 0.41991, -0.73905, 0.54086],
                             [0.99495, 0.44499, 0.36828, -0.8086],
                             [1.0008, -0.38781, 0.91443, 0.13808]])
        else:
            raise ValueError('PMP temperature not supported')

    elif detector == 'HRT':
        if temperature == 40:
            O = np.array([[0.99816, 0.61485, 0.010613, -0.77563],
                          [0.99192, 0.08382, 0.86254, 0.46818],
                          [1.0042, -0.84437, 0.12872, -0.53972],
                          [1.0057, -0.30576, -0.87969, 0.40134]])

        elif temperature == 50:
            O = np.array([[1.0014, 0.56715, 0.3234, -0.74743],
                          [1.007, 0.0037942, 0.69968, 0.71423],
                          [1.002, -0.98937, 0.04716, -0.20392],
                          [0.99769, 0.27904, -0.86715, 0.39908]])
        else:
            raise ValueError('PMP temperature not supported')
    else:
        raise ValueError('Detector not supported')

    M = rotation_matrix(angle)
    return np.matmul(O, M)


def rotation_matrix(angle_rot):
    c, s = np.cos(2 * angle_rot * np.pi / 180), np.sin(2 * angle_rot * np.pi / 180)
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])
