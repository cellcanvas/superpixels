import pandas as pd
from numpy.typing import ArrayLike
from skimage.measure import regionprops_table
import numpy as np

def intensity_voxel_counts(
        regionmask: ArrayLike, 
        intensity_image: ArrayLike
) -> pd.DataFrame:
    # Define the custom property function to count the number of voxels for each intensity

    # Ensure intensity_image contains integers
    intensity_image = intensity_image.astype(np.int64)
    
    # Apply the region mask to the intensity image
    region_intensity_values = intensity_image[regionmask]

    # Count number of label pixels per integer label
    counts = np.bincount(region_intensity_values, minlength=8)  
    
    return counts

##
# Function to calculate stats for each intensity
#
# Returns a list of stats:
# labels_present_count,
# labels_present_count/total_labels,
# majority_label_count/total,
# background_count/majority_label_count,
# background_count,
# majority_label_count,
# total,
#
def intensity_voxel_stats(
        regionmask: ArrayLike,
        intensity_image: ArrayLike
) -> pd.DataFrame:
    # Define the custom property function to calc stats for each intensity
    #    [0] = number of (non-background) labels present
    #    [1] = ratio labelled pixels to all pixels (0-1)
    #    [2] = ratio majority labelled pixels  to all pixels (0-1)
    #    [3] = ratio background pixels to majority labelled pixels (0-1)


    # Ensure intensity_image contains integers
    intensity_image = intensity_image.astype(np.int64)

    # Apply the region mask to the intensity image
    region_intensity_values = intensity_image[regionmask]

    # Count number of label pixels per integer label
    counts = np.bincount(region_intensity_values, minlength=8)

    if (non_zero_voxel_counts(counts)):
        background_count = counts[0]

        total = float(region_intensity_values.shape[0])  # Avoid integer division
        total_labels = float(counts.shape[0]) # Avoid integer division

        majority_label = np.argmax(counts[1:])+1  # Skip the background which is at index 0
        majority_label_count = counts[majority_label]
        labels_present_count = np.count_nonzero(counts[1:])

        stats = [
            labels_present_count,
            labels_present_count/total_labels,
            majority_label_count/total,
            background_count/majority_label_count,
            background_count,
            majority_label_count,
            total,
        ]
        return stats
    else:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def non_zero_voxel_counts(counts: ArrayLike) -> bool:
    for count in counts[1:]:
        if count > 0:
            return True
    return False

def get_gt_label_per_super_pixel(row) :
    # Function to from the 8 label counts to single label (the class with most pixels, or background)

    counts = row.values

    weights = np.ones_like(counts,dtype='float')
    weights[0] = 0.3

    idx = np.argmax(counts * weights)

    # # if at least pixel in the superpixel has a gt-label, assign this label (1-7)
    # if np.max(counts[1:])>0:
    #     idx = np.argmax(counts[1:])+1
    # # if no gt-label is present in superpixel, assign background (0)
    # else:
    #     idx = 0
    return idx

def ground_truth_count(
    superpixels: ArrayLike,
    ground_truth: ArrayLike,
) -> pd.DataFrame:
    
    # count number of pixels per 'ground_truth'-label per superpixel
    props = regionprops_table(
        superpixels,
        intensity_image=ground_truth,
        properties=['label'],  # Add other properties as needed
        extra_properties=[intensity_voxel_counts]
        )
    
    # make into dataframe and set label as index
    props_df = pd.DataFrame(props)
    props_df = props_df.set_index('label')

    # get majority label per superpixel and set column name
    gt_df = props_df.apply(get_gt_label_per_super_pixel, axis=1)
    gt_df = gt_df.to_frame()
    gt_df.columns = ['ground_truth']

    return gt_df

def ground_truth_stats(
        superpixels: ArrayLike,
        ground_truth: ArrayLike,
) -> pd.DataFrame:

    # count number of pixels per 'ground_truth'-label per superpixel
    props = regionprops_table(
        superpixels,
        intensity_image=ground_truth,
        properties=['label'],  # Add other properties as needed
        extra_properties=[intensity_voxel_stats]
    )

    # make into dataframe and set label as index
    props_df = pd.DataFrame(props, dtype=float)
    props_df = props_df.set_index('label')

    props_df.columns = [
        "labels_present_count",
        "labels_present_count/total_labels",
        "majority_label_count/total",
        "background_count/majority_label_count",
        "background_count",
        "majority_label_count",
        "total",
    ]

    return props_df
