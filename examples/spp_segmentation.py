import os
import zarr
import napari
import numpy as np
import copick
import click
import copick_utils
from magicgui import magicgui
from napari.types import ImageData, LabelsData
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


@magicgui(auto_call=True, sigma={"widget_type": "FloatSlider", "min": 0, "max": 10}, h_minima={"widget_type": "FloatSlider", "min": 0, "max": 50})
def _spp_widget(image: ImageData, sigma: float = 4, h_minima: float = 0.001) -> LabelsData:
    return superpixels(image, sigma=sigma, h_minima=h_minima)

# REF: https://github.com/pyapp-kit/magicgui/issues/226
_spp_widget.h_minima._widget._readout_widget.setDecimals(4)


@click.command()
@click.option("--interactive", type=bool, is_flag=True, default=False, help="Run in interactive mode")
def main(interactive: bool) -> None:
    """
    Run the superpixel segmentation on a tomogram from the copick dataset.
    Usage:
        python spp_segmentation.py <OPTIONAL --interactive>
    """

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

    if interactive:
        # cropping when interactive, because segmentation is slow
        img = img[50:100, 180:360, 210:430]

    print("Loading image into memory ...")
    img = np.asarray(img) # Loading into memory

    print("Segmenting superpixels ...")
    segm = superpixels(img, sigma=4, h_minima=0.0025)
    print("Done ...")

    # Open viewer and add image
    viewer = napari.Viewer()
    viewer.add_image(img)
    viewer.add_labels(segm)

    if interactive:
        viewer.window.add_dock_widget(_spp_widget, area="right")

    # uncomment to save the segmentation
    # imwrite("segm.tif", segm)

    napari.run()


if __name__ == "__main__":
    main()
