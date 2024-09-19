from typing import Callable, Optional, Dict, Union

import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from scipy import ndimage
from torch.utils.data import Dataset


class SuperpixelsDataset(Dataset):
    def __init__(
        self,
        image: ArrayLike,
        superpixels: ArrayLike,
        labels: pd.Series,
        transform: Optional[Callable] = None,
        remove_background: bool = False,
    ) -> None:
        super().__init__()

        self.image = image
        self.superpixels = superpixels
        self.labels = labels
        self.remove_background = remove_background
        self.transform = transform

        self.bboxes = {
            spp: bbox 
            for spp, bbox in enumerate(ndimage.find_objects(superpixels), start=1)
            if bbox is not None
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, Union[ArrayLike, int]]:

        label = self.labels.iloc[idx]
        spp_idx = self.labels.index[idx]

        bbox = self.bboxes[spp_idx]
        crop = self.image[bbox]

        if self.remove_background:
            bkg_mask = self.superpixels[bbox] != spp_idx
            crop[bkg_mask] = 0
        
        data = {
            "image": crop,
            "label": label,
        }

        if self.transform:
            data = self.transform(data)
        
        return data
