from typing import Callable, Optional, Dict, Union

import pandas as pd
from numpy.typing import ArrayLike
from scipy.ndimage import find_objects
from torch.utils.data import Dataset


class SuperpixelsDataset(Dataset):
    def __init__(
        self,
        image: ArrayLike,
        superpixels: ArrayLike,
        labels: pd.Series,
        transforms: Optional[Callable] = None,
        remove_background: bool = False,
    ) -> None:
        super().__init__()

        self.image = image
        self.superpixels = superpixels
        self.labels = labels
        self.remove_background = remove_background
        self.transforms = transforms

        self.bboxes = {
            spp: bbox 
            for spp, bbox in enumerate(find_objects(superpixels), start=1)
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
        
        if self.transforms:
            crop = self.transforms(crop)
        
        return {
            "image": crop,
            "label": label,
        }
