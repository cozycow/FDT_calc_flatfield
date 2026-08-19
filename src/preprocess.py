from os import path
import numpy as np
from astropy.io import fits
from prefilter_correction import correct_prefilter
from cavity_correction import correct_cavity
from utils import *
from scipy.ndimage import gaussian_filter
from limb_fitting import find_center


def preprocess(file,
               dark_file=None,
               prefilter_file=None,
               cavity_file=None,
               flatfield_file=None,
               deadpix_file=None,
               ghost_file=None,
               distortion_file=None,
               #true_continuum=False,
               calc_dc=False,
               folder_out='',
               to_file=False,
               verbose=True):

    with fits.open(file) as hdul:
        data = hdul[0].data
        header_data = hdul[0].header
        img_data = hdul['PHI_FITS_imageSummary'].data

    cpos = int(header_data['CONTPOS']) - 1
    scale_data = get_scale(img_data)
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

    if prefilter_file is not None:
        data = correct_prefilter(data, header_data, prefilter_file)

    if cavity_file is not None:
        with fits.open(cavity_file) as hdul:
            cavity = hdul[0].data
        data = correct_cavity(data, header_data, cavity)

    if flatfield_file is not None:
        with fits.open(flatfield_file) as hdul:
            flat = hdul[0].data
        data = data / crop(flat, header_data)

    if deadpix_file is not None:
        with fits.open(deadpix_file) as hdul:
            deadpix = hdul[0].data[:, ::-1].astype(bool) ###
        deadpix = crop(deadpix, header_data)
        data[...,~deadpix] = np.nan
        data = fill_holes(data)
        data = np.nan_to_num(data)

    if ghost_file is not None:
        with fits.open(ghost_file) as hdul:
            ghost = hdul[0].data
        xr, yr = reflection_point_predict(header_data)
        reflection = reflect(gaussian_filter(data[cpos,0], 8), xr, yr)
        data -= reflection * crop(ghost, header_data)

    if distortion_file is not None:
        s = np.load(distortion_file)
        xd, yd = s['xd'], s['yd']
        data = undistort(data, header_data, xd, yd)

    #if true_continuum:
    #    data[cpos] = calc_continuum(data, header_data)

    if calc_dc:
        xc, yc, rsun = find_center(data[cpos,0])
        header_data['CRPIX2'] = round(xc + 1, 4)
        header_data['CRPIX1'] = round(yc + 1, 4)
        header_data['CDELT1'] = round(header_data['RSUN_ARC'] / rsun, 6)
        header_data['CDELT2'] = round(header_data['RSUN_ARC'] / rsun, 6)

    if to_file:
        file_out = path.join(folder_out, generate_filename(file))
        clone_fits(file, file_out, data.astype(np.float32), header_data)
    else:
        return data.astype(np.float32), header_data