import numpy as np
from scipy.ndimage import map_coordinates


def undistort(data, header, xd, yd, **kwargs):
    if len(data.shape) == 2:
        xd_, yd_ = crop_grid(xd, yd, header)
        return map_coordinates(data, (xd_, yd_), **kwargs)
    else:
        out = []
        for i in range(len(data)):
            out.append(undistort(data[i], header, xd, yd, **kwargs))
        return np.array(out)


def crop_grid(xi, yi, header):
    nx, ny = header['NAXIS2'], header['NAXIS1']
    x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1
    return xi[x0:x0 + nx, y0:y0 + ny] - x0, yi[x0:x0 + nx, y0:y0 + ny] - y0
