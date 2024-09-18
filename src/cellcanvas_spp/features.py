import pandas as pd
from numpy.typing import ArrayLike
from skimage.measure import regionprops_table
import numpy as np
from tqdm import tqdm

def superpixel_regionprops_features(
    image: ArrayLike,
    superpixels: ArrayLike,
) -> pd.DataFrame:
    properties = [
        'label',
        'area',
        'bbox',
        'bbox_area',
        'centroid',
        'equivalent_diameter',
        'euler_number',
        'extent',
        'filled_area',
        'major_axis_length',
        'max_intensity',
        'mean_intensity',
        'min_intensity',
        'std_intensity',
    ]
    df = pd.DataFrame(
        regionprops_table(
            label_image=superpixels,
            intensity_image=image,
            properties=properties,
        )
    )
    return  df


def superpixel_cellcanvas_features(
    embeddings: ArrayLike,
    superpixels: ArrayLike,
    embedding_axis: int = 0,
) -> pd.DataFrame:
    """
    Retun the median of the embeddings for each superpixel.
    """
    if embedding_axis != len(embeddings.shape) - 1:
        embeddings = np.moveaxis(embeddings, embedding_axis, -1)
    
    dt = dict()
    labels = np.unique(superpixels)
    for label in tqdm(labels):
        ssp= np.median(embeddings[superpixels==label], axis=0)
        dt[label] = ssp

    return  pd.DataFrame(dt)