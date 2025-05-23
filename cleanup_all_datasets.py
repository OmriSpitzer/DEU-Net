from pathlib import Path

def cleanup_images(images_dir, masks_dir, image_ext, mask_ext, mask_suffix=None):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    if mask_suffix:
        mask_basenames = set(mask.stem.replace(mask_suffix, '') for mask in masks_dir.glob(f'*{mask_suffix}.{mask_ext}'))
    else:
        mask_basenames = set(mask.stem for mask in masks_dir.glob(f'*.{mask_ext}'))
    for image in images_dir.glob(f'*.{image_ext}'):
        if image.stem not in mask_basenames:
            print(f"Deleting {image}")
            image.unlink()

def main():
    # ISIC2016
    cleanup_images(
        'datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_Data',
        'datasets/ISIC2016/ISBI2016_ISIC_Part1_Training_GroundTruth',
        'jpg', 'png', '_Segmentation'
    )
    cleanup_images(
        'datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_Data',
        'datasets/ISIC2016/ISBI2016_ISIC_Part1_Test_GroundTruth',
        'jpg', 'png', '_Segmentation'
    )
    # ISIC2017
    cleanup_images(
        'datasets/ISIC2017/ISIC-2017_Training_Data',
        'datasets/ISIC2017/ISIC-2017_Training_Part1_GroundTruth',
        'jpg', 'png'
    )
    cleanup_images(
        'datasets/ISIC2017/ISIC-2017_Validation_Data',
        'datasets/ISIC2017/ISIC-2017_Validation_Part1_GroundTruth',
        'jpg', 'png'
    )
    cleanup_images(
        'datasets/ISIC2017/ISIC-2017_Test_v2_Data',
        'datasets/ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth',
        'jpg', 'png'
    )
    # ISIC2018
    cleanup_images(
        'datasets/ISIC2018/ISIC2018_Task1-2_Training_Input',
        'datasets/ISIC2018/ISIC2018_Task1_Training_GroundTruth',
        'jpg', 'png'
    )
    # PH2
    ph2_images_dir = Path('datasets/PH2/PH2 Dataset images')
    for case_dir in ph2_images_dir.iterdir():
        img_dir = case_dir / f"{case_dir.name}_Dermoscopic_Image"
        mask_dir = case_dir / f"{case_dir.name}_lesion"
        cleanup_images(img_dir, mask_dir, 'bmp', 'bmp', None)

if __name__ == "__main__":
    main() 