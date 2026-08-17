import numpy as np


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


def roll(data, dx, dy, cval=0):
    nx, ny = data.shape[-2:]
    dx_, dy_ = int(round(dx)), int(round(dy))
    data_ = np.zeros_like(data) + cval
    data_[...,max(dx_,0):min(nx+dx_,nx), max(dy_,0):min(ny+dy_,ny)] = data[...,max(-dx_,0):min(nx-dx_,nx), max(-dy_,0):min(ny-dy_,ny)].copy()
    return data_


def forward(datas, shifts, weights, reference, **kwargs):
    for data, shift, weight in zip(datas, shifts, weights):
        yield (roll(data, *(shifts[0] - shift), **kwargs),
               roll(reference, *(shifts[0] - shift), **kwargs),
               roll(weight, *(shifts[0] - shift), **kwargs))

def backward(datas, shifts, weights, reference, **kwargs):
    for data, shift, weight in zip(datas, shifts, weights):
        yield (data,
               roll(reference, *(shift - shifts[0]), **kwargs),
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

