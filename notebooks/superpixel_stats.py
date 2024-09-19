# %%
# The following magic causes the notebook to reload external python modules upon execution of a cell
# This is useful when developing modules in parallel to the notebook

import copick
from pathlib import Path
from cellcanvas_spp.ground_truth import copick_to_ground_truth_image, ground_truth_stats

import numpy as np
import pandas as pd
from tifffile import imread

# %%
DATA_DIR = Path('/Users/jordao.bragantini/Softwares/superpixels/notebooks/data/copick_10439')

root = copick.from_file(DATA_DIR / "synthetic_data_10439_dataportal.json")

runs = ["16193", "16191"]

stats = []

for run in runs:
    gt = copick_to_ground_truth_image(root, run)
    segm = imread(DATA_DIR / f"segm_{run}.tif")
    import napari
    viewer = napari.view_labels(segm)
    viewer.add_labels(gt)

    df = ground_truth_stats(segm, gt)
    df["run"] = run
    stats.append(df)

df = pd.concat(stats)


# %%

print(df.describe())


# %%
