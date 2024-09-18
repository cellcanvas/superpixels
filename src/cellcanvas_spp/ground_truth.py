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


def get_gt_label_per_super_pixel(
        df: pd.DataFrame,
) -> pd.DataFrame:
    # Function to from the 8 label counts to single label (the class with most pixels, or background)

    max_label = np.zeros(df.shape[0])
    for index, row in df.iterrows():
        counts = row.values
        # if at least pixel in the superpixel has a gt-label, assign this label (1-7)
        if np.max(counts[1:])>0:
            idx = np.argmax(counts[1:])+1
        # if no gt-label is present in superpixel, assign background (0)
        else:
            idx = 0
        max_label[index] = idx
    df_out = pd.DataFrame(max_label, columns=['ground_truth'])

    return df_out


def ground_truth_count(
    superpixels: ArrayLike,
    ground_truth: ArrayLike,
) -> pd.DataFrame:
    
    # count number of pixels per 'ground_truth'-label per superpixel
    props = regionprops_table(
        superpixels,
        intensity_image=ground_truth,
        # properties=['label', 'area'],  # Add other properties as needed
        extra_properties=[intensity_voxel_counts]
        )

    # make dataframe with 8 columns (1 per label)
    props_df = pd.DataFrame(props)

    # remove unnecessary columns
    props_df = props_df.drop(['label', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'bbox-4', 'bbox-5'], axis=1)

    # rename columns
    for i in range(8):
        props_df = props_df.rename(columns={"intensity_voxel_counts-"+str(i): "label_"+str(i)})

    # collapse 8 columns to 1 columns (based on majority vote)
    gt_df = get_gt_label_per_super_pixel(props_df)

    return gt_df