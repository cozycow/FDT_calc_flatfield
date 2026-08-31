import numpy as np
from scipy.ndimage import gaussian_filter


def correct_ghost(data, header, ghost, **kwargs):
    '''
    :param data: numpy array of shape (24,nx,ny) or (6,4,nx,ny) containing modulated intensities
    :param header: fits header
    :param ghost: ghost map of shape (4,nx,ny)
    :return: corrected data
    '''

    xr, yr = reflection_point_predict(header)
    data_ = data.copy().reshape((6, -1, data.shape[-2], data.shape[-1]))

    reflection = data_[:,:1].copy()
    reflection = gaussian_filter(reflection, 8, axes=(-2, -1))
    reflection = gaussian_filter(reflection, 1., axes=(0,), mode='wrap')
    reflection = reflect(reflection, xr, yr)
    data_ -= reflection * np.expand_dims(crop(ghost, header), 0)

    return data_.reshape(data.shape)


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image


def roll_float(data, dx, dy, **kwargs):
    from scipy.ndimage import map_coordinates

    if len(data.shape) == 2:
        nx, ny = data.shape
        xi, yi = np.mgrid[:nx,:ny].astype(np.float32)
        xi -= dx
        yi -= dy
        return map_coordinates(data, (xi, yi), **kwargs)
    else:
        out = []
        for i in range(len(data)):
            out.append(roll_float(data[i], dx, dy, **kwargs))
        return np.array(out)


def reflect(data, xr, yr, **kwargs):
    nx, ny = data.shape[-2:]
    return roll_float(data[...,::-1, ::-1], 2 * int(round(xr)) - nx + 1, 2 * int(round(yr)) - ny + 1, **kwargs)


def reflection_point_predict(header):
    px = [1.63114715e-06, 6.72511045e-03, 9.60448053e+02]
    py = [ 4.61830880e-06, -6.85005911e-03,  9.77508840e+02]

    r_sun = header['RSUN_ARC']
    dx, dy = header['PXBEG2'] - 1, header['PXBEG1'] - 1

    xr = np.polyval(px, r_sun) - dx
    yr = np.polyval(py, r_sun) - dy
    return xr, yr
