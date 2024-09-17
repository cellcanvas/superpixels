import os
import zarr
import napari
import numpy as np
import copick
import copick_utils
from tifffile import imwrite
from pathlib import Path

from cellcanvas_spp.segmentation import superpixels


try:
    DATA_DIR = Path(os.environ["COPICK_DATA"])
except KeyError:
    raise ValueError(
        "Please set the COPICK_DATA environment variable to point to the data directory\n\n"
        "$ export COPICK_DATA=</path/to/copick/data> python <script>"
    )


def main() -> None:

    # Use copick
    # config_file = "./copick_10301/synthetic_data_10301_dataportal.json"
    config_file = DATA_DIR / "copick_10439/synthetic_data_10439_dataportal.json"
    root = copick.from_file(config_file)

    # run_name = "14075"
    run_name = "16193"
    print(root)
    example_run = root.get_run(run_name)
    voxel_spacing= 10
    tomo_type = "wbp"
    tomogram = example_run.voxel_spacings[0].tomograms[0]

    # Open zarr
    z = zarr.open(tomogram.zarr())
    img = z["0"] # Get our highest resolution scale

    print("Loading image into memory ...")
    img = np.asarray(img) # Loading into memory

    print("Segmenting superpixels ...")
    segm = superpixels(img, sigma=4, h_minima=0.0025)
    print("Done ...")

    # uncomment to save the segmentation
    # imwrite("segm.tif", segm)

    # Open viewer and add image
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_labels(segm)

    napari.run()



if __name__ == "__main__":
    main()
