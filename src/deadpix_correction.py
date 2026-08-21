import numpy as np
from skimage.restoration import inpaint
from scipy.ndimage import binary_fill_holes


def correct_deadpix(data, header, deadpix):
    deadpix = crop(deadpix, header)
    data[..., ~deadpix] = np.nan
    data = fill_holes(data)
    return np.nan_to_num(data)


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


def crop(image, header=None, x1=None, x2=None, y1=None, y2=None, **kwargs):
    if header is not None:
        x1, x2, y1, y2 = header['PXBEG2'] - 1, header['PXEND2'], header['PXBEG1'] - 1, header['PXEND1']
    nx, ny = x2 - x1 + 1, y2 - y1 + 1

    if (isinstance(image, np.ndarray) and (len(image.shape) > 1) and (image.shape[-2:] != (nx, ny)) and
            x1 is not None and x2 is not None and y1 is not None and y2 is not None):
        return image[..., x1:x2, y1:y2]
    else:
        return image
