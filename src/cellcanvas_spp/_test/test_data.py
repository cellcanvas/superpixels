from skimage.data import cells3d
from cellcanvas_spp.data import SuperpixelsDataset

import scipy.ndimage as ndi
import numpy as np
import pandas as pd


def test_data_loader() -> None:

    rng = np.random.default_rng(42)

    image = cells3d()[:, 1]

    smooth = ndi.gaussian_filter(image, sigma=1)
    superpixels, n_spps = ndi.label(smooth > smooth.mean())

    labels = rng.integers(0, 3, size=n_spps)
    labels = pd.Series(labels, index=np.unique(superpixels)[1:])  # ignoring 0

    dataset = SuperpixelsDataset(image, superpixels, labels, remove_background=True)

    for img, lb in dataset:
        # qualitative check :D
        print(img.shape)
        print(lb)
