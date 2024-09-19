import os
import zarr
import napari
import numpy as np
import copick
import copick_utils
from magicgui import magicgui
from napari.types import ImageData, LabelsData
from pathlib import Path
from cellcanvas_spp.segmentation import superpixels
import pandas as pd
from skimage.measure import regionprops_table
from skimage.morphology import ball, disk
import matplotlib.cm as cm
import matplotlib.colors as colors

try:
    # DATA_DIR = Path("/Users/kharrington/git/cellcanvas/superpixels/notebooks/my_synthetic_data_10439_dataportal.json")
    DATA_DIR = Path(os.environ["COPICK_DATA"])
except KeyError:
    raise ValueError(
        "Please set the COPICK_DATA environment variable to point to the data directory\n\n"
        "$ export COPICK_DATA=</path/to/copick/data>"
    )

@magicgui(auto_call=True, sigma={"widget_type": "FloatSlider", "min": 0, "max": 10}, h_minima={"widget_type": "FloatSlider", "min": 0, "max": 50})
def _spp_widget(image: ImageData, sigma: float = 4, h_minima: float = 0.001) -> LabelsData:
    return superpixels(image, sigma=sigma, h_minima=h_minima)

_spp_widget.h_minima._widget._readout_widget.setDecimals(4)

def get_segmentation_array(copick_run, segmentation_name, voxel_spacing=10, is_multilabel=True):
    seg = copick_run.get_segmentations(is_multilabel=is_multilabel, name=segmentation_name, voxel_size=voxel_spacing)
    if len(seg) == 0:
        return None
    segmentation = zarr.open(seg[0].zarr().path, mode="r")['0']
    return segmentation[:]

def segment_superpixels(example_run, voxel_spacing=10, interactive: bool = False):
    """
    This function handles the segmentation logic. If the segmentation exists in copick, it will be skipped.
    """
    segmentation_name = "superpixelSegmentation"

    # Check if the segmentation already exists
    seg = get_segmentation_array(example_run, segmentation_name, voxel_spacing=voxel_spacing, is_multilabel=True)
    if seg is not None:
        print(f"Segmentation '{segmentation_name}' already exists. Skipping segmentation.")
        return seg

    # Proceed with superpixel segmentation if it does not exist
    tomo_type = "wbp"
    tomogram = example_run.voxel_spacings[0].tomograms[0]

    # Open zarr
    z = zarr.open(tomogram.zarr())
    img = z["0"]  # Get the highest resolution scale

    if interactive:
        img = img[50:100, 180:360, 210:430]  # Cropping for interactive mode

    print("Loading image into memory ...")
    img = np.asarray(img)  # Loading into memory

    print("Segmenting superpixels ...")
    segm = superpixels(img, sigma=4, h_minima=0.0025)
    print("Done ...")

    # Save segmentation into copick
    print("Saving segmentation to copick...")
    new_seg = example_run.new_segmentation(voxel_spacing, segmentation_name, session_id="0", is_multilabel=True, user_id="cellcanvasSPP")
    segmentation_group = zarr.open_group(new_seg.path, mode="a")
    segmentation_group["0"] = segm
    print("Segmentation saved.")

    return segm

def prepare_and_run_segmentation(interactive: bool = False):
    """
    Prepare and run the segmentation by fetching data from copick.
    """
    config_file = DATA_DIR
    root = copick.from_file(config_file)

    run_name = "16193"
    example_run = root.get_run(run_name)
    voxel_spacing = 10

    # Check for existing segmentation or run segmentation if not found
    seg = segment_superpixels(example_run, voxel_spacing, interactive)

    # Open viewer and add image
    viewer = napari.Viewer()
    img = np.asarray(zarr.open(example_run.voxel_spacings[0].tomograms[0].zarr())["0"])  # Load image
    viewer.add_image(img, name='Image')

    # Compute a comprehensive set of region properties for the segmentation labels
    print("Computing region properties for segmentation...")
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
    props = regionprops_table(seg, intensity_image=img, properties=properties)
    props_df = pd.DataFrame(props)
    print("Comprehensive region properties computed.")

    # Initialize the 'painted_label' column with zeros (unpainted)
    props_df['painted_label'] = 0

    # Add the labels layer with features
    label_layer = viewer.add_labels(
        seg,
        name="Superpixels",
        features=props_df,
    )

    # Function to update color mapping based on 'painted_label'
    def update_color_mapping():
        painted_labels = label_layer.features['painted_label'].values
        labels = label_layer.features['label'].values

        # Identify painted labels (painted_labels != 0)
        painted_mask = painted_labels != 0
        painted_labels_filled = painted_labels[painted_mask]
        labels_painted = labels[painted_mask]

        if painted_labels_filled.size == 0:
            return  # No painted labels to update

        unique_painted_labels = np.unique(painted_labels_filled)

        # Create a colormap
        norm = colors.Normalize(vmin=unique_painted_labels.min(), vmax=unique_painted_labels.max())
        colormap = cm.ScalarMappable(norm=norm, cmap='viridis')

        # Map 'painted_label' to colors
        painted_label_to_color = {
            painted_label: colormap.to_rgba(painted_label)
            for painted_label in unique_painted_labels
        }

        # Map label ids to colors
        label_color_mapping = {}
        for label_id, painted_label in zip(labels, painted_labels):
            if painted_label != 0:
                label_color_mapping[label_id] = painted_label_to_color[painted_label]
            else:
                label_color_mapping[label_id] = (0, 0, 0, 0)  # Transparent color for unpainted labels

        # Update the labels layer with the new color mapping
        label_layer.color_mode = 'direct'
        label_layer.color = label_color_mapping

    # Initial color mapping
    update_color_mapping()

    # Define the custom painting function
    def custom_paint(layer, event):
        """Custom painting function that updates features DataFrame and color mapping."""
        # On press
        _update_paint(layer, event)
        yield

        # On move
        while event.type == 'mouse_move':
            _update_paint(layer, event)
            yield

    def _update_paint(layer, event):
        # Get the coordinates in data space
        data_coordinates = layer.world_to_data(event.position).astype(int)
        # Get the brush size
        brush_size = layer.brush_size
        # Get the brush footprint
        if layer.ndim == 3:
            brush_radius = int(np.round(brush_size / 2))
            brush_footprint = ball(brush_radius)
        else:
            brush_radius = int(np.round(brush_size / 2))
            brush_footprint = disk(brush_radius)
        # Get the indices of the footprint
        offsets = np.argwhere(brush_footprint) - brush_radius
        # Get the coordinates being painted
        coords_painted = data_coordinates + offsets
        # Ensure the coordinates are within the image bounds
        coords_painted = coords_painted[
            np.all((coords_painted >= 0) & (coords_painted < np.array(layer.data.shape)), axis=1)
        ]
        # Now, get the previous labels at those coordinates
        coords_tuple = tuple(coords_painted.T)
        prev_labels = layer.data[coords_tuple]
        # Now, get the unique previous labels
        unique_prev_labels = np.unique(prev_labels)
        # Now, paint the new label at those coordinates
        layer.data[coords_tuple] = layer.selected_label
        # Now, for each unique previous label, update the features DataFrame directly
        for prev_label in unique_prev_labels:
            # Get the index positions where 'label' equals 'prev_label'
            idx = label_layer.features['label'].values == prev_label
            # Modify 'painted_label' directly using the underlying NumPy array
            label_layer.features['painted_label'].values[idx] = layer.selected_label

        # Now, update the color mapping
        update_color_mapping()

    # Connect the custom painting function to the labels layer
    label_layer.mouse_drag_callbacks.append(custom_paint)

    # Add ground truth
    base_seg = np.zeros_like(zarr.open(example_run.segmentations[0].zarr())["0"])
    for idx, segm in enumerate(example_run.segmentations):
        z = zarr.open(segm.zarr())["0"][:]
        base_seg = base_seg + (idx + 1) * z
    viewer.add_labels(base_seg, name="Ground Truth")

    # If interactive mode, show the widget
    if interactive:
        viewer.window.add_dock_widget(_spp_widget, area="right")

    # Return the viewer for further interaction
    return viewer

# Run segmentation directly
viewer = prepare_and_run_segmentation(interactive=False)
napari.run()
