import numpy as np
from astropy.io import fits
from scipy.ndimage import map_coordinates
from skimage.restoration import inpaint
from scipy.ndimage import binary_fill_holes


def read_wavelengths(header):
    nwv = header['WAVENUM']
    wv = []
    for i in range(nwv):
        wv.append(header[f'WAVELN{i + 1:02d}'])
    return np.array(wv)


def clone_fits(file_in, file_out, data):
    with fits.open(file_in) as hdul:
        hdul[0].data = data.astype(np.float32)
        hdul.writeto(file_out, overwrite=True)


def generate_filename(file, prefix='ilam', extension='.fits'):
    from datetime import datetime

    temp = file.split('/')[-1].split('.')[0].split('_')
    return '_'.join(['-'.join(temp[2].split('-')[:2]) + '-' + prefix, temp[3],
                     'V' + datetime.today().strftime('%Y%m%d%H%M') + temp[4][-1],  temp[-1]]) + extension


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image


def rebin(data, k, axis=None):
    if len(data.shape) == 2:
        nx, ny = data.shape
        if axis == 0:
            return np.mean(np.reshape(data[:nx // k * k, :], (nx // k, -1, ny)), axis=-2)
        elif axis == 1:
            return np.mean(np.reshape(data[:, :ny // k * k], (nx, ny // k, -1)), axis=-1)
        else:
            return rebin(rebin(data, k, axis=0), k, axis=1)
    else:
        out = []
        for i in range(len(data)):
            out.append(rebin(data[i], k, axis=axis))
        return np.array(out)


def undistort(data, header, xd, yd, **kwargs):
    def crop_grid(xi, yi, header):
        nx, ny = header['NAXIS2'], header['NAXIS1']
        x0, y0 = header['PXBEG2'] - 1, header['PXBEG1'] - 1
        return xi[x0:x0 + nx, y0:y0 + ny] - x0, yi[x0:x0 + nx, y0:y0 + ny] - y0

    if len(data.shape) == 2:
        xd_, yd_ = crop_grid(xd, yd, header)
        return map_coordinates(data, (xd_, yd_), **kwargs)
    else:
        out = []
        for i in range(len(data)):
            out.append(undistort(data[i], header, xd, yd, **kwargs))
        return np.array(out)


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


def reflect(data, xr, yr, **kwargs):
    nx, ny = data.shape[-2:]
    return roll_float(data[...,::-1, ::-1], 2 * int(round(xr)) - nx + 1, 2 * int(round(yr)) - ny + 1, **kwargs)


def realign(data, x0=None, y0=None, **kwargs):
    from limb_fitting import find_center
    data_ = data.copy().reshape((-1, data.shape[-2], data.shape[-1]))

    if x0 is None and y0 is None:
        x0, y0, _ = find_center(data_[0], **kwargs)

    for i in range(len(data_)):
        xc, yc, _ = find_center(data_[i], **kwargs)
        dx, dy = x0 - xc, y0 - yc
        data_[i] = roll_float(data_[i], dx, dy)

    return data_.reshape(data.shape)


def fill_holes(data):
    if len(data.shape) == 2:
        mask = np.isnan(data)
        not_holes = binary_fill_holes(~mask)
        holes = not_holes & mask
        image_ = np.nan_to_num(data, nan=0)
        image_ = inpaint.inpaint_biharmonic(image_, holes)
        image_[~not_holes] = np.nan
        return image_
    else:
        out = []
        for i in range(len(data)):
            out.append(fill_holes(data[i]))
        return np.array(out)


def remove_freq(image, kx, ky, h, thr, expand=False, nx0=2048, ny0=2048, **kwargs):
    nx, ny = image.shape
    image_ = np.where(np.isnan(image) + (np.abs(image) > thr), 0, image)
    high = image - image_

    # expand the image to the full size
    if expand:
        temp = np.zeros((nx0, ny0)) + np.nanmedian(image)
        temp[:nx, :ny] = image_
        image_ = temp

    nx_, ny_ = image_.shape
    kx_, ky_ = (np.round(np.array(kx) * nx_ / nx0).astype(int) + nx_ // 2,
                np.round(np.array(ky) * ny_ / ny0).astype(int) + ny_ // 2)
    h_ = int(np.ceil(h * nx_ / nx0))

    # apply Fourier transform and shift the zero-frequency component to the center of the spectrum
    fft = np.fft.fft2(image_)
    fft = np.fft.fftshift(fft)

    # remove the frequencies and their negative counterparts
    for kxi, kyi in zip(kx_, ky_):
        fft[kxi - h_:kxi + h_ + 1, kyi - h_:kyi + h_ + 1] = 0
        fft[-kxi - h_:-kxi + h_ + 1, -kyi - h_:-kyi + h_ + 1] = 0

    # shift the zero frequency back and apply inverse Fourier transform
    fft = np.fft.ifftshift(fft)
    fft = np.fft.ifft2(fft)
    return np.real(fft)[:nx, :ny] + high


def calc_continuum(data, x, n_comp=101, sigma=0.04, gamma=0.05, continuum=-1, lam=1e-6, shift=0, return_coeff=False, **kwargs):
    from scipy.special import voigt_profile

    n = len(x)
    x_min = np.min(np.delete(x, continuum) if continuum is not None else x) + shift
    x_max = np.max(np.delete(x, continuum) if continuum is not None else x) + shift
    dx = (x_max - x_min) / (n_comp - 1)
    xc = (x_min + x_max) / 2
    x_ = np.arange(x_min, x_max + dx / 2, dx, dtype=np.float32)

    A = voigt_profile(np.expand_dims(x, axis=1) - np.expand_dims(x_, axis=0), sigma, gamma, dtype=np.float32)
    A0 = np.mean(A, axis=0, keepdims=True)
    A = A - A0

    W = np.diag(voigt_profile(x_ - xc, sigma, gamma) ** 2)

    q = 1 / n - A0 @ W @ A.T @ np.linalg.inv(A @ W @ A.T + lam * np.identity(n)) @ (np.identity(n) - 1 / n)

    if return_coeff:
        return q[0]
    else:
        nx, ny = data.shape[-2:]
        return np.linalg.tensordot(data.reshape(n,-1,nx,ny), q[0], axes=(0, 0))


def polyterms2d(x, y, degree=1):
    if degree == 0:
        return np.array([np.ones_like(x)])
    else:
        return np.append(polyterms2d(x, y, degree=degree-1), np.array([x ** (degree - i) * y ** i for i in range(degree + 1)]), axis=0)


def polyval2d(x, y, p):
    degree = int((np.sqrt(8 * len(p) + 1) - 3) / 2)
    return np.sum([p_ * x_ for p_, x_ in zip(p, polyterms2d(x, y, degree=degree))], axis=0)


def polyfit2d(f, x=None, y=None, degree=1, weight=None, return_coefficients=False):
    if x is None and y is None:
        nx, ny = f.shape
        x, y = np.mgrid[-nx // 2 + 0.5:nx // 2 + 0.5, -ny // 2 + 0.5:ny // 2 + 0.5].astype(np.float32)
        x /= nx / 2
        y /= ny / 2

    if weight is None:
        W = np.ones_like(f)
    else:
        W = weight

    t = np.where(~np.isnan(f))
    X = polyterms2d(x[t], y[t], degree=degree)[1:]
    Y = f[t]
    W = W[t]
    W /= np.sum(W)

    X0 = np.sum(X * W, axis=-1, keepdims=True)
    Y0 = np.sum(Y * W)

    X_ = X - X0
    Y_ = Y - Y0

    k = (Y_ * W) @ X_.T @ np.linalg.inv((X_ * W) @ X_.T)
    b = Y0 - k @ X0
    p = np.append(b, k)

    if return_coefficients:
        return p
    else:
        return polyval2d(x, y, p)


def modulation_matrix(temperature=45):
    if temperature == 45:
        return np.array([[1.0023, -0.64814, -0.56202, -0.51859],
                        [1.0041, 0.54693, -0.55299, 0.633],
                        [0.99523, 0.46132, 0.54165, -0.69603],
                        [0.99838, -0.61944, 0.66189, 0.42519]])
    else: #temperature == 40
        return np.array([[0.99913, -0.69504, -0.38074, -0.60761],
                         [1.0051, 0.41991, -0.73905, 0.54086],
                         [0.99495, 0.44499, 0.36828, -0.8086],
                         [1.0008, -0.38781, 0.91443, 0.13808]])


def modulate(data, temperature=45, inv=False):
    O = modulation_matrix(temperature)

    if inv:
        O = np.linalg.inv(O)

    nx, ny = data.shape[-2:]
    data_ = data.copy().reshape((-1,4,nx,ny)).transpose((2,3,1,0))
    data_ = O @ data_
    return data_.transpose((3,2,0,1)).reshape(data.shape)


def demodulate(data, temperature=45):
    return modulate(data, temperature=temperature, inv=True)


def reflection_point_predict(header):
    px = [1.63114715e-06, 6.72511045e-03, 9.60448053e+02]
    py = [ 4.61830880e-06, -6.85005911e-03,  9.77508840e+02]

    r_sun = header['RSUN_ARC']
    dx, dy = header['PXBEG2'] - 1, header['PXBEG1'] - 1

    xr = np.polyval(px, r_sun) - dx
    yr = np.polyval(py, r_sun) - dy
    return xr, yr


def interpolate(f, x, x_new):
    idx = np.searchsorted(x, x_new).clip(1, len(x) - 1)
    xa, xb = x[idx - 1], x[idx]
    dx = xb - xa

    a, b = (xb - x_new) / dx, (x_new - xa) / dx
    fa = np.take_along_axis(f, idx - 1, axis=0)
    fb = np.take_along_axis(f, idx, axis=0)
    return fa * a + fb * b
