import os
from pathlib import Path

# Paths
images_dir = Path('datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_Data')
masks_dir = Path('datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_GroundTruth')

# Get all mask base names (without _Segmentation)
mask_basenames = set(mask.stem.replace('_Segmentation', '') for mask in masks_dir.glob('*_Segmentation.png'))

# Delete images without a corresponding mask
for image in images_dir.glob('*.jpg'):
    if image.stem not in mask_basenames:
        print(f"Deleting {image}")
        image.unlink() 