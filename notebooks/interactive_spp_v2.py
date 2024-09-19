# %%
# painting superpixels

import os
import shutil
import zarr
import napari
import numpy as np
import pandas as pd
import copick
from pathlib import Path
from cellcanvas_spp.segmentation import superpixels
from skimage.measure import regionprops_table
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from skimage import future
from functools import partial
import threading
import toolz as tz
from psygnal import debounced
from superqt import ensure_main_thread
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QWidget,
)
from appdirs import user_data_dir
import logging
import tifffile

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up the data directory
DATA_DIR = Path("/Users/jordao.bragantini/Softwares/superpixels/notebooks/data/copick_10439/synthetic_data_10439_dataportal.json")
run_name = "16191"

# Load the tomogram
def load_tomogram():
    config_file = DATA_DIR
    root = copick.from_file(config_file)
    example_run = root.get_run(run_name)
    tomogram = example_run.voxel_spacings[0].tomograms[0]
    z = zarr.open(tomogram.zarr())
    img = z["0"]  # Get the highest resolution scale
    return np.asarray(img)

# Load and crop the tomogram
full_tomogram = load_tomogram()
# crop_3D = full_tomogram[50:100, 180:360, 210:430]  # Adjust crop as needed
crop_3D = full_tomogram[:]

# Compute superpixels
# superpixel_seg = superpixels(crop_3D, sigma=4, h_minima=0.0025)
superpixel_seg = tifffile.imread(DATA_DIR.parent / f'segm_{run_name}_10000.tif')

# Set up Napari viewer
viewer = napari.Viewer()
scale = (1, 1, 1)  # Adjust scale if needed
contrast_limits = (crop_3D.min(), crop_3D.max())

# Add layers
data_layer = viewer.add_image(crop_3D, scale=scale, contrast_limits=contrast_limits, name="Tomogram")
superpixel_layer = viewer.add_labels(superpixel_seg, scale=scale, name="Superpixels", opacity=0.5)

# Set up zarr for prediction and painting layers
zarr_path = os.path.join(user_data_dir("napari_dl_at_mbl_2024", "napari"), "diy_segmentation.zarr")
print(f"zarr path: {zarr_path}")
shutil.rmtree(zarr_path, ignore_errors=True)
prediction_data = zarr.open(f"{zarr_path}/prediction", mode='a', shape=crop_3D.shape, dtype='i4', dimension_separator="/")
painting_data = zarr.open(f"{zarr_path}/painting", mode='a', shape=crop_3D.shape, dtype='i4', dimension_separator="/")

prediction_layer = viewer.add_labels(prediction_data, name="Prediction", scale=scale)
painting_layer = viewer.add_labels(painting_data, name="Painting", scale=scale)

# Precompute regionprops features for each superpixel
def compute_superpixel_features(image, superpixels):
    # df_path = DATA_DIR.parent / f'{run_name}_embeddings.csv'
    df_path = DATA_DIR.parent / f'{run_name}_regionprops.csv'

    if df_path.exists():
        props = pd.read_csv(df_path)
        try:
            props.drop(columns=["umap_x", "umap_y"], inplace=True)
        except:
            pass
        assert "label" in props.columns
    else:
        print(df_path, "NOT FOUND!!!")
        props = regionprops_table(superpixels, intensity_image=image,
                                properties=('label', 
                                            'area', 
                                            # 'bbox',
                                            # 'bbox_area',
                                            # 'centroid',
                                            'equivalent_diameter',
                                            'euler_number',
                                            # 'extent',
                                            'filled_area',
                                            'major_axis_length',
                                            'max_intensity',
                                            'mean_intensity',
                                            'min_intensity',
                                            'std_intensity',))
    return props

superpixel_features = compute_superpixel_features(crop_3D, superpixel_seg)

def update_model(y, X, model_type):
    logger.debug(f"X shape: {X.shape}")
    logger.debug(f"y shape: {y.shape}")
    logger.debug(f"Unique labels: {np.unique(y)}")
    
    if y.size == 0:
        logger.warning("No labeled data found. Skipping model update.")
        return None
    
    if model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05, class_weight='balanced')
    
    try:
        clf.fit(X, y)
        logger.info("Model successfully updated")
        return clf
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        return None

def predict(model, superpixel_features):
    features = np.array([[superpixel_features[prop][i] for prop in superpixel_features.keys() if prop != 'label'] 
                         for i in range(len(superpixel_features['label']))])
    prediction = model.predict(features)
    return prediction

# Napari ML Widget
class NapariMLWidget(QWidget):
    def __init__(self, parent=None):
        super(NapariMLWidget, self).__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        model_label = QLabel("Select Model")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["Random Forest"])
        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        layout.addLayout(model_layout)

        self.live_fit_checkbox = QCheckBox("Live Model Fitting")
        self.live_fit_checkbox.setChecked(True)
        layout.addWidget(self.live_fit_checkbox)

        self.live_pred_checkbox = QCheckBox("Live Prediction")
        self.live_pred_checkbox.setChecked(True)
        layout.addWidget(self.live_pred_checkbox)

        self.setLayout(layout)

# Add widget to Napari
widget = NapariMLWidget()
viewer.window.add_dock_widget(widget, name="Interactive Segmentation")

# Event listener
model = None

@tz.curry
def on_data_change(event, viewer=None, widget=None):
    painting_layer.refresh()

    thread = threading.Thread(
        target=threaded_on_data_change,
        args=(
            event,
            viewer.dims,
            widget.model_dropdown.currentText(),
            widget.live_fit_checkbox.isChecked(),
            widget.live_pred_checkbox.isChecked(),
        ),
    )
    thread.start()
    thread.join()

    prediction_layer.refresh()

def threaded_on_data_change(
    event,
    dims,
    model_type,
    live_fit,
    live_prediction,
):
    global model, crop_3D, painting_data, superpixel_seg, superpixel_features
    
    # Ensure consistent shapes
    min_shape = [min(s1, s2, s3) for s1, s2, s3 in zip(crop_3D.shape, painting_data.shape, superpixel_seg.shape)]
    logger.debug(f"min_shape: {min_shape}")
    
    active_labels = painting_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    logger.debug("active_labels")
    crop_3D_subset = crop_3D[:min_shape[0], :min_shape[1], :min_shape[2]]
    logger.debug("crop subset")
    superpixel_seg_subset = superpixel_seg[:min_shape[0], :min_shape[1], :min_shape[2]]
    logger.debug("superpixel subset")

    # Recompute superpixel features
    # logger.debug(f"computing superpixel shapes")
    # superpixel_features = compute_superpixel_features(crop_3D_subset, superpixel_seg_subset)
    
    # Create a mask of painted pixels
    painted_mask = active_labels > 0
    
    logger.debug("painted mask")

    if live_fit:
        logger.debug("preparing live fit")
        
        # Create a mask of painted pixels
        painted_mask = active_labels > 0
        
        # Get unique superpixel labels in the painted areas
        painted_superpixels = np.unique(superpixel_seg_subset[painted_mask])
        
        # Prepare features and labels for training
        X = []
        y = []
        
        for label in painted_superpixels:
            mask = superpixel_seg_subset == label
            painted_pixels = active_labels[mask & painted_mask]
            
            if painted_pixels.size > 0:
                feature_vector = [superpixel_features[prop][superpixel_features['label'] == label].iloc[0] 
                                for prop in superpixel_features.keys() if prop != 'label']
                X.append(feature_vector)
                y.append(stats.mode(painted_pixels, axis=None)[0])

        X = np.array(X)
        y = np.array(y)
        logger.debug("data prepared for live fit")
        
        logger.debug(f"Number of painted superpixels: {len(X)}")
        logger.debug(f"X shape: {X.shape}, y shape: {y.shape}")
        
        if len(X) > 0:
            model = update_model(y, X, model_type)
        else:
            logger.warning("No painted superpixels found. Skipping model update.")

    if live_prediction and model is not None:
        try:
            logger.debug("Starting prediction")
            
            # Prepare features for all superpixels
            features = np.array([
                [superpixel_features[prop][i] for prop in superpixel_features.keys() if prop != 'label']
                for i in range(len(superpixel_features['label']))
            ])
            # features = superpixel_features.drop(columns='label').to_numpy()
            
            logger.debug("Starting actual prediction")
            # Predict for all superpixels
            superpixel_predictions = model.predict(features)
            
            logger.debug("Creating mapping")
            # Create a mapping from superpixel label to prediction
            label_to_prediction = dict(zip(superpixel_features['label'], superpixel_predictions))
            
            # Use numpy vectorize to apply the mapping efficiently
            prediction_func = np.vectorize(lambda x: label_to_prediction.get(x, 0))
            prediction = prediction_func(superpixel_seg_subset)
            
            # Ensure prediction has the correct shape and dtype
            prediction = prediction.astype(prediction_layer.data.dtype)
            
            logger.debug("Updating layer")
            # Update the prediction layer data
            prediction_layer.data[:min_shape[0], :min_shape[1], :min_shape[2]] = prediction
            
            logger.debug("Prediction updated successfully")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.exception("Detailed traceback:")

# Connect event listeners
for listener in [painting_layer.events.paint]:
    listener.connect(
        debounced(
            ensure_main_thread(
                on_data_change(viewer=viewer, widget=widget)
            ),
            timeout=1000,
        )
    )

napari.run()

# %%
superpixel_features.keys()


# %%



