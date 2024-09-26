import os
from pathlib import Path
import zarr
import numpy as np
from tqdm import tqdm
import copick
from skimage.feature import multiscale_basic_features
from cellcanvas_spp.segmentation import superpixels
import pickle
from tifffile import imread

try:
    DATA_DIR = Path(os.environ["COPICK_DATA"])
except KeyError:
    raise ValueError(
        "Please set the COPICK_DATA environment variable to point to the data directory\n\n"
        "$ export COPICK_DATA=</path/to/copick/data> python <script>"
    )

# config_file = DATA_DIR / "copick_10439/synthetic_data_10439_dataportal.json"
config_file = "synthetic_data_10439.json"
root = copick.from_file(config_file)

particles = dict()
for po in root.config.pickable_objects:
    particles[po.name] = po.label


list_of_runs = [root.get_run(s) for s in ["16191", ]]   #"16193","16191"

data_dict = {}
for run in tqdm(list_of_runs):

    print(f"Preparing run {run.name}")
    tomogram = run.get_voxel_spacing(10).get_tomogram('wbp')
    _, array = list(zarr.open(tomogram.zarr()).arrays())[0]
    tomogram = array[:]
    mask = np.zeros(tomogram.shape)
    segmentations = run.get_segmentations()

    # print("Calculating SK features...")
    # sk_features = multiscale_basic_features(
    #         tomogram,
    #         intensity=True,
    #         edges=True,
    #         texture=True,
    #         sigma_min=0.5,
    #         sigma_max=8.0
    #     )
    #sk_features = np.moveaxis(features, -1, 0)

    # print("Calculating superpixels...")
    # segm = superpixels(tomogram, sigma=4, h_minima=0.0005)

    print("Loading superpixels...")
    segm = imread('data/copick_10439/segm_' + run.name + '_10000.tif')      #CHANGEEEE

    print("Converting ground-truth labels...")
    for seg in segmentations:
        _, array = list(zarr.open(seg.zarr()).arrays())[0]
        arr = np.array(array[:])
        mask[arr==1] = particles[seg.name]
    
    data_dict = {"image": tomogram, 
                 "label": mask, 
                #  "sk_features": sk_features,
                 "superpixels": segm}
    
    print("Saving data to pickle file...")
    with open(f'data/copick_10439/dataset_run_{run.name}_loadedSegm10000.pickle', 'wb') as f:  # 'wb' means write in binary mode
        pickle.dump(data_dict, f)
 
