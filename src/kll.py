import numpy as np
from scipy.ndimage import map_coordinates


def kll(datas, shifts, weights,
        niter=20, sigma=None,
        vmin=None, vmax=None,
        slope=False,
        **kwargs):

    if slope:
        F = np.ones_like(datas[0])
    else:
        F = np.zeros_like(datas[0])

    for iter in range(niter):
        D = step(forward(datas, shifts, weights, F, **kwargs), slope=slope)
        F = step(backward(datas, shifts, weights, D, **kwargs), slope=slope, guess=F, sigma=sigma).clip(vmin, vmax)

    return F


def roll_float(data, dx, dy, **kwargs):
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


def forward(datas, shifts, weights, reference, **kwargs):
    for data, shift, weight in zip(datas, shifts, weights):
        yield (roll_float(data, *(shifts[0] - shift), **kwargs),
               roll_float(reference, *(shifts[0] - shift), **kwargs),
               roll_float(weight, *(shifts[0] - shift), **kwargs))

def backward(datas, shifts, weights, reference, **kwargs):
    for data, shift, weight in zip(datas, shifts, weights):
        yield (data,
               roll_float(reference, *(shift - shifts[0]), **kwargs),
               weight)

def step(data, slope=False, guess=None, sigma=None):
    with np.errstate(invalid='ignore'):
        A, B, W = 0, 0, 0
        for y, x, w_ in data:
            if guess is not None and sigma is not None:
                if slope:
                    r = y - x * guess
                else:
                    r = y - x - guess

                w = w_ * sigma / np.abs(r).clip(sigma)
            else:
                w = w_

            W += w
            if slope:
                A += np.nan_to_num((y * x - A) * w / W)
                B += np.nan_to_num((x ** 2 - B) * w / W)
            else:
                A += np.nan_to_num((y - A) * w / W)
                B += np.nan_to_num((x - B) * w / W)

        if slope:
            return np.nan_to_num(A / B)
        else:
            return A - B

