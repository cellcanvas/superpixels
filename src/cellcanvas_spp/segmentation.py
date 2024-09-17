
from numpy.typing import ArrayLike
from pyift.shortestpath import watershed_from_minima
from skimage.filters import sobel

import scipy.ndimage as ndi


def superpixels(
    image: ArrayLike,
    sigma: ArrayLike,
    h_minima: float,
) -> ArrayLike:
    """
    Segmentation of an image into superpixels using the watershed algorithm.
    The image is smoothed with a Gaussian filter and the gradient is computed
    using the Sobel operator.

    Parameters
    ----------
    image : ArrayLike
        Input image.
    sigma : ArrayLike
        Standard deviation for Gaussian smoothing.
    h_minima : float
        Height used to filter out non-significant minima from gradient image.
    
    Returns
    -------
    ArrayLike
        Segmented image.
    """
    smooth_img = ndi.gaussian_filter(image, sigma=sigma)
    grad = sobel(smooth_img)

    _, segm = watershed_from_minima(grad, H_minima=h_minima)

    return segm
