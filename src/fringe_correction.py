import numpy as np


def correct_fringes(data, thr=3e-3, window_size=3,
                    kx=(3, 15, 12, 18, 22),
                    ky=(27, 27, 0, 20, 16)):
    '''
    :param data: numpy array of shape (24,nx,ny) or (6,4,nx,ny) containing demodulated data
    :param kx: tuple of ints, horizontal frequencies to be removed
    :param ky: tuple of ints, vertical frequencies to be removed
    :param h: int, frequency peak width
    :param thr: float, clipping threshold
    :return: corrected Stokes parameters
    '''

    data_ = data.copy().reshape((-1, 4, data.shape[-2], data.shape[-1]))

    for j in range(data_.shape[0]):  # loop over wavelengths
        for i in range(1,4):
            temp = data_[j, i] / data_[j, 0].clip(1)
            temp = temp - remove_freq(temp, kx, ky, window_size, thr=thr)
            data_[j, i] -= temp * data[j, 0]

    return data_.reshape(data.shape)


def remove_freq(image, kx, ky, h, thr, nx0=2048, ny0=2048, **kwargs):
    nx, ny = image.shape
    image_ = np.where(np.isnan(image) + (np.abs(image) > thr), 0, image)
    high = image - image_

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
