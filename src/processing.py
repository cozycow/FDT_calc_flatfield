from os import path
import numpy as np
from astropy.io import fits
from prefilter_correction import correct_prefilter
from cavity_correction import correct_cavity
from ghost_correction import correct_ghost
from fringe_correction import correct_fringes
from crosstalk_correction import correct_crosstalk
from deadpix_correction import correct_deadpix
from limb_fitting import find_center, realign
from distortion_correction import undistort
from modulation import demodulate
from wavelengths import get_wavelengths


def process(file,
            dark_file=None,
            prefilter_file=None,
            cavity_file=None,
            flatfield_file=None,
            deadpix_file=None,
            ghost_file=None,
            distortion_file=None,
            _find_center=False,
            _realign=False,
            _demodulate=False,
            _correct_fringes=False,
            _correct_crosstalk=False,
            folder_out='',
            to_file=False,
            verbose=True):

    with fits.open(file) as hdul:
        data = hdul[0].data
        header_data = hdul[0].header
        img_data = hdul['PHI_FITS_imageSummary'].data
        fg_data = hdul['PHI_FITS_FG_settings'].data

    if 'WAVENUM' not in header_data:
        _ = get_wavelengths(header_data, fg_data, update_header=True)
    scale_data = get_scale(img_data)

    cpos = int(header_data['CONTPOS']) - 1
    detector_data = header_data['DETECTOR']

    nx, ny = data.shape[-2:]
    data = data.reshape(6, 4, nx, ny)

    if dark_file is not None:
        with fits.open(dark_file) as hdul:
            dark = hdul[0].data
            header_dark = hdul[0].header
            img_dark = hdul['PHI_FITS_imageSummary'].data

        scale_dark = get_scale(img_dark)
        detector_dark = header_dark['DETECTOR']
        if detector_dark != detector_data:
            dark = dark[:,::-1]
        data -= scale_data / scale_dark * crop(dark, header_data)

    if flatfield_file is not None:
        with fits.open(flatfield_file) as hdul:
            flat = hdul[0].data
        data = data / crop(flat, header_data)

    if prefilter_file is not None:
        data = correct_prefilter(data, header_data, prefilter_file)

    if ghost_file is not None:
        with fits.open(ghost_file) as hdul:
            ghost = hdul[0].data
        data = correct_ghost(data, header_data, ghost)

    if cavity_file is not None:
        with fits.open(cavity_file) as hdul:
            cavity = hdul[0].data
        data = correct_cavity(data, header_data, cavity)

    if deadpix_file is not None:
        with fits.open(deadpix_file) as hdul:
            deadpix = hdul[0].data[:, ::-1].astype(bool) ###
        data = correct_deadpix(data, header_data, deadpix)

    if distortion_file is not None:
        s = np.load(distortion_file)
        xd, yd = s['xd'], s['yd']
        data = undistort(data, header_data, xd, yd)

    if _realign:
        data = realign(data)

    if _find_center:
        xc, yc, rsun = find_center(data[cpos,0])
        header_data['CRPIX2'] = round(xc + 1, 4)
        header_data['CRPIX1'] = round(yc + 1, 4)
        header_data['CDELT1'] = round(header_data['RSUN_ARC'] / rsun, 6)
        header_data['CDELT2'] = round(header_data['RSUN_ARC'] / rsun, 6)

    if _demodulate:
        data = demodulate(data, header_data)

    if _correct_fringes:
        data = correct_fringes(data)

    if _correct_crosstalk:
        data = correct_crosstalk(data, header_data)

    if to_file:
        file_out = path.join(folder_out, generate_filename(file))
        clone_fits(file, file_out, data.astype(np.float32), header_data)
    else:
        return data.astype(np.float32), header_data


def clone_fits(file_in, file_out, data, header=None):
    with fits.open(file_in) as hdul:
        hdul[0].data = data.astype(np.float32)
        if header is not None:
            hdul[0].header = header
        hdul.writeto(file_out, overwrite=True)


def get_scale(img_data):
    if img_data is not None:
        fmt, rng = img_data['PHI_IMG_format'], img_data['PHI_IMG_maxRange']

        scale = rng[-1] / rng[0]
        if fmt[-1] != 'IMGFMT_24_8':
            scale *= 256
        return scale
    else:
        return None


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
