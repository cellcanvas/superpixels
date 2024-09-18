from typing import Literal

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


def superpixels_hws(
    image: ArrayLike,
    sigma: float = 2,
    area_threshold: int = 10_000,
    watershed: Literal["dynamics", "area", "volume"] = "dynamics",
    weight_function: Literal["min", "mean", "max"] = "mean", 
) -> ArrayLike:
    """
    Segmentation of an image into superpixels using the hierarchical watershed algorithm.
    The image is smoothed with a Gaussian filter and the gradient is computed
    using the Sobel operator.

    Parameters
    ----------
    image : ArrayLike
        Input image.
    sigma : float, optional
        Standard deviation for Gaussian smoothing, by default 2.
    area_threshold : int, optional
        Maximum area of superpixels, by default 10_000.
    watershed : Literal["dynamics", "area", "volume"], optional
        Watershed criterium, by default "dynamics".
    weight_function : Literal["min", "mean", "max"], optional
        Weight function, by default "mean".
    
    Returns
    -------
    ArrayLike
        Superpixels labels.
    """
    try:
        import higra as hg
    except ImportError:
        raise ImportError(
            "higra is required to run this function.\n"
            "conda install -c conda-forge higra\n"
            "It isn't available on PyPI for all platforms."
        )

    # compute gradient for watershed
    smooth = ndi.gaussian_filter(image, sigma=sigma)
    grad = sobel(smooth)

    # create graph
    mask = [[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]
    neighbours = hg.mask_2_neighbours(mask)
    graph = hg.get_nd_regular_graph(image.shape, neighbours)

    # compute graph edge weights
    weight_func = getattr(hg.WeightFunction, weight_function)
    edge_weights = hg.weight_graph(graph, grad, weight_function=weight_func)

    # compute hierarchical watershed
    ws_func = getattr(hg, f"watershed_hierarchy_by_{watershed}")
    tree, alt = ws_func(graph, edge_weights)

    # cutting the hierarchy by area
    area = hg.attribute_area(tree)
    labels = hg.labelisation_horizontal_cut_from_threshold(tree, area, area_threshold)

    return labels
