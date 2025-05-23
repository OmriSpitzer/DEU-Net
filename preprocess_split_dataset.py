import random

from pathlib import Path


def isic2016(dataset_dir, output_dir, image_ext='jpg', mask_ext='png',
             val_ratio=0.125, split_random=True, random_seed=None):
    # The validation and training data are obtained by randomly splitting the original training data

    if random_seed is not None:
        random.seed(random_seed)

    if val_ratio > 1:
        raise ValueError("val_ratio is too big.")

    dataset_dir = Path(dataset_dir)

    train_val_images = list((dataset_dir / 'ISBI2016_ISIC_Part1_Training_Data').glob(f'*.{image_ext}'))
    train_val_masks = list((dataset_dir / 'ISBI2016_ISIC_Part1_Training_GroundTruth').glob(f'*_Segmentation.{mask_ext}'))

    print(f"Found {len(train_val_images)} images and {len(train_val_masks)} masks in training data.")

    # Create a mapping of image names to mask names
    image_to_mask = {}
    for mask_path in train_val_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        image_to_mask[image_name] = mask_path

    # Match images with their corresponding masks
    train_val_data = []
    unmatched_images = []
    for image_path in train_val_images:
        image_name = image_path.stem
        if image_name in image_to_mask:
            train_val_data.append((image_path, image_to_mask[image_name]))
        else:
            unmatched_images.append(image_path.name)

    unmatched_masks = []
    image_names = set(img.stem for img in train_val_images)
    for mask_path in train_val_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        if image_name not in image_names:
            unmatched_masks.append(mask_path.name)

    print(f"Matched {len(train_val_data)} image-mask pairs.")
    if unmatched_images:
        print(f"Images without masks: {unmatched_images}")
    if unmatched_masks:
        print(f"Masks without images: {unmatched_masks}")

    if len(train_val_data) == 0:
        raise ValueError("No matched image-mask pairs found in training data.")

    val_count = int(len(train_val_data) * val_ratio)
    train_count = len(train_val_data) - val_count

    if split_random:
        random.shuffle(train_val_data)

    train_data = train_val_data[:train_count]
    val_data = train_val_data[train_count:]

    test_images = list((dataset_dir / 'ISBI2016_ISIC_Part1_Test_Data').glob(f'*.{image_ext}'))
    test_masks = list((dataset_dir / 'ISBI2016_ISIC_Part1_Test_GroundTruth').glob(f'*_Segmentation.{mask_ext}'))

    print(f"Found {len(test_images)} test images and {len(test_masks)} test masks.")

    # Create a mapping of test image names to mask names
    test_image_to_mask = {}
    for mask_path in test_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        test_image_to_mask[image_name] = mask_path

    # Match test images with their corresponding masks
    test_data = []
    unmatched_test_images = []
    for image_path in test_images:
        image_name = image_path.stem
        if image_name in test_image_to_mask:
            test_data.append((image_path, test_image_to_mask[image_name]))
        else:
            unmatched_test_images.append(image_path.name)

    unmatched_test_masks = []
    test_image_names = set(img.stem for img in test_images)
    for mask_path in test_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        if image_name not in test_image_names:
            unmatched_test_masks.append(mask_path.name)

    print(f"Matched {len(test_data)} test image-mask pairs.")
    if unmatched_test_images:
        print(f"Test images without masks: {unmatched_test_images}")
    if unmatched_test_masks:
        print(f"Test masks without images: {unmatched_test_masks}")

    if len(train_val_images) != len(train_val_masks) or len(train_val_images) == 0:
        raise ValueError("The set of images is empty or the number of masks is not equal to the number of images.")

    __save_output(output_dir, train_data, val_data, test_data)


def isic2016_224x224(dataset_dir, output_dir, image_ext='png', mask_ext='png',
                     val_ratio=0.125, split_random=True, random_seed=None):
    # The validation and training data are obtained by randomly splitting the original training data

    if random_seed is not None:
        random.seed(random_seed)

    if val_ratio > 1:
        raise ValueError("val_ratio is too big.")

    dataset_dir = Path(dataset_dir)

    train_val_images = list((dataset_dir / 'ISBI2016_ISIC_Part1_Training_Data').glob(f'*.{image_ext}'))
    train_val_masks = list((dataset_dir / 'ISBI2016_ISIC_Part1_Training_GroundTruth').glob(f'*_Segmentation.{mask_ext}'))
    print(f"Found {len(train_val_images)} images and {len(train_val_masks)} masks in ISIC2016_224x224 training data.")

    # Create a mapping of image names to mask names
    image_to_mask = {}
    for mask_path in train_val_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        image_to_mask[image_name] = mask_path

    # Match images with their corresponding masks
    train_val_data = []
    unmatched_images = []
    for image_path in train_val_images:
        image_name = image_path.stem
        if image_name in image_to_mask:
            train_val_data.append((image_path, image_to_mask[image_name]))
        else:
            unmatched_images.append(image_path.name)

    unmatched_masks = []
    image_names = set(img.stem for img in train_val_images)
    for mask_path in train_val_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        if image_name not in image_names:
            unmatched_masks.append(mask_path.name)

    print(f"Matched {len(train_val_data)} image-mask pairs in ISIC2016_224x224.")
    if unmatched_images:
        print(f"Images without masks in ISIC2016_224x224: {unmatched_images}")
    if unmatched_masks:
        print(f"Masks without images in ISIC2016_224x224: {unmatched_masks}")

    if len(train_val_data) == 0:
        raise ValueError("No matched image-mask pairs found in ISIC2016_224x224 training data.")

    val_count = int(len(train_val_data) * val_ratio)
    train_count = len(train_val_data) - val_count

    if split_random:
        random.shuffle(train_val_data)

    train_data = train_val_data[:train_count]
    val_data = train_val_data[train_count:]

    test_images = list((dataset_dir / 'ISBI2016_ISIC_Part1_Test_Data').glob(f'*.{image_ext}'))
    test_masks = list((dataset_dir / 'ISBI2016_ISIC_Part1_Test_GroundTruth').glob(f'*_Segmentation.{mask_ext}'))
    print(f"Found {len(test_images)} test images and {len(test_masks)} test masks in ISIC2016_224x224.")

    # Create a mapping of test image names to mask names
    test_image_to_mask = {}
    for mask_path in test_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        test_image_to_mask[image_name] = mask_path

    # Match test images with their corresponding masks
    test_data = []
    unmatched_test_images = []
    for image_path in test_images:
        image_name = image_path.stem
        if image_name in test_image_to_mask:
            test_data.append((image_path, test_image_to_mask[image_name]))
        else:
            unmatched_test_images.append(image_path.name)

    unmatched_test_masks = []
    test_image_names = set(img.stem for img in test_images)
    for mask_path in test_masks:
        image_name = mask_path.stem.replace('_Segmentation', '')
        if image_name not in test_image_names:
            unmatched_test_masks.append(mask_path.name)

    print(f"Matched {len(test_data)} test image-mask pairs in ISIC2016_224x224.")
    if unmatched_test_images:
        print(f"Test images without masks in ISIC2016_224x224: {unmatched_test_images}")
    if unmatched_test_masks:
        print(f"Test masks without images in ISIC2016_224x224: {unmatched_test_masks}")

    if len(test_data) == 0:
        raise ValueError("No matched image-mask pairs found in ISIC2016_224x224 test data.")

    __save_output(output_dir, train_data, val_data, test_data)


def isic2017(dataset_dir, output_dir, image_ext='jpg', mask_ext='png'):
    dataset_dir = Path(dataset_dir)

    train_images = list((dataset_dir / 'ISIC-2017_Training_Data').glob(f'*.{image_ext}'))
    train_masks = list((dataset_dir / 'ISIC-2017_Training_Part1_GroundTruth').glob(f'*.{mask_ext}'))
    print(f"Found {len(train_images)} training images and {len(train_masks)} training masks in ISIC2017.")

    # Create a mapping of image names to mask names
    image_to_mask = {}
    for mask_path in train_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        image_to_mask[image_name] = mask_path

    # Match images with their corresponding masks
    train_data = []
    unmatched_train_images = []
    for image_path in train_images:
        image_name = image_path.stem
        if image_name in image_to_mask:
            train_data.append((image_path, image_to_mask[image_name]))
        else:
            unmatched_train_images.append(image_path.name)

    unmatched_train_masks = []
    image_names = set(img.stem for img in train_images)
    for mask_path in train_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        if image_name not in image_names:
            unmatched_train_masks.append(mask_path.name)

    print(f"Matched {len(train_data)} training image-mask pairs in ISIC2017.")
    if unmatched_train_images:
        print(f"Training images without masks in ISIC2017: {unmatched_train_images}")
    if unmatched_train_masks:
        print(f"Training masks without images in ISIC2017: {unmatched_train_masks}")
    if len(train_data) == 0:
        raise ValueError("No matched image-mask pairs found in ISIC2017 training data.")

    val_images = list((dataset_dir / 'ISIC-2017_Validation_Data').glob(f'*.{image_ext}'))
    val_masks = list((dataset_dir / 'ISIC-2017_Validation_Part1_GroundTruth').glob(f'*.{mask_ext}'))
    print(f"Found {len(val_images)} validation images and {len(val_masks)} validation masks in ISIC2017.")

    # Create a mapping of validation image names to mask names
    val_image_to_mask = {}
    for mask_path in val_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        val_image_to_mask[image_name] = mask_path

    # Match validation images with their corresponding masks
    val_data = []
    unmatched_val_images = []
    for image_path in val_images:
        image_name = image_path.stem
        if image_name in val_image_to_mask:
            val_data.append((image_path, val_image_to_mask[image_name]))
        else:
            unmatched_val_images.append(image_path.name)

    unmatched_val_masks = []
    val_image_names = set(img.stem for img in val_images)
    for mask_path in val_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        if image_name not in val_image_names:
            unmatched_val_masks.append(mask_path.name)

    print(f"Matched {len(val_data)} validation image-mask pairs in ISIC2017.")
    if unmatched_val_images:
        print(f"Validation images without masks in ISIC2017: {unmatched_val_images}")
    if unmatched_val_masks:
        print(f"Validation masks without images in ISIC2017: {unmatched_val_masks}")
    if len(val_data) == 0:
        raise ValueError("No matched image-mask pairs found in ISIC2017 validation data.")

    test_images = list((dataset_dir / 'ISIC-2017_Test_v2_Data').glob(f'*.{image_ext}'))
    test_masks = list((dataset_dir / 'ISIC-2017_Test_v2_Part1_GroundTruth').glob(f'*.{mask_ext}'))
    print(f"Found {len(test_images)} test images and {len(test_masks)} test masks in ISIC2017.")

    # Create a mapping of test image names to mask names
    test_image_to_mask = {}
    for mask_path in test_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        test_image_to_mask[image_name] = mask_path

    # Match test images with their corresponding masks
    test_data = []
    unmatched_test_images = []
    for image_path in test_images:
        image_name = image_path.stem
        if image_name in test_image_to_mask:
            test_data.append((image_path, test_image_to_mask[image_name]))
        else:
            unmatched_test_images.append(image_path.name)

    unmatched_test_masks = []
    test_image_names = set(img.stem for img in test_images)
    for mask_path in test_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        if image_name not in test_image_names:
            unmatched_test_masks.append(mask_path.name)

    print(f"Matched {len(test_data)} test image-mask pairs in ISIC2017.")
    if unmatched_test_images:
        print(f"Test images without masks in ISIC2017: {unmatched_test_images}")
    if unmatched_test_masks:
        print(f"Test masks without images in ISIC2017: {unmatched_test_masks}")
    if len(test_data) == 0:
        raise ValueError("No matched image-mask pairs found in ISIC2017 test data.")

    __save_output(output_dir, train_data, val_data, test_data)


def isic2018(dataset_dir, output_dir, image_ext='jpg', mask_ext='png',
             train_ratio=0.7, val_ratio=0.1, split_random=True, random_seed=None):
    # we only use the training data of this dataset!

    if random_seed is not None:
        random.seed(random_seed)

    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio is too big.")

    dataset_dir = Path(dataset_dir)

    train_val_test_images = list((dataset_dir / 'ISIC2018_Task1-2_Training_Input').glob(f'*.{image_ext}'))
    train_val_test_masks = list((dataset_dir / 'ISIC2018_Task1_Training_GroundTruth').glob(f'*.{mask_ext}'))
    print(f"Found {len(train_val_test_images)} images and {len(train_val_test_masks)} masks in ISIC2018.")

    # Create a mapping of image names to mask names (remove _segmentation)
    image_to_mask = {}
    for mask_path in train_val_test_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        image_to_mask[image_name] = mask_path

    # Match images with their corresponding masks
    train_val_test_data = []
    unmatched_images = []
    for image_path in train_val_test_images:
        image_name = image_path.stem
        if image_name in image_to_mask:
            train_val_test_data.append((image_path, image_to_mask[image_name]))
        else:
            unmatched_images.append(image_path.name)

    unmatched_masks = []
    image_names = set(img.stem for img in train_val_test_images)
    for mask_path in train_val_test_masks:
        image_name = mask_path.stem.replace('_segmentation', '')
        if image_name not in image_names:
            unmatched_masks.append(mask_path.name)

    print(f"Matched {len(train_val_test_data)} image-mask pairs in ISIC2018.")
    if unmatched_images:
        print(f"Images without masks in ISIC2018: {unmatched_images}")
    if unmatched_masks:
        print(f"Masks without images in ISIC2018: {unmatched_masks}")
    if len(train_val_test_data) == 0:
        raise ValueError("No matched image-mask pairs found in ISIC2018.")

    train_count = int(len(train_val_test_data) * train_ratio)
    val_count = int(len(train_val_test_data) * val_ratio)

    if split_random:
        random.shuffle(train_val_test_data)

    train_data = train_val_test_data[:train_count]
    val_data = train_val_test_data[train_count:train_count + val_count]
    test_data = train_val_test_data[train_count + val_count:]

    __save_output(output_dir, train_data, val_data, test_data)


def ph2(dataset_dir, output_dir, image_ext='bmp', mask_ext='bmp',
        train_ratio=0.7, val_ratio=0.1, split_random=True, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio is too big.")

    dataset_dir = Path(dataset_dir)

    train_val_test_images = []
    train_val_test_masks = []

    for tmp_path in (dataset_dir / 'PH2 Dataset images').iterdir():
        train_val_test_images.append(
            tmp_path / ("%s_Dermoscopic_Image" % tmp_path.name) / f"{tmp_path.name}.{image_ext}")
        train_val_test_masks.append(tmp_path / ("%s_lesion" % tmp_path.name) / f"{tmp_path.name}_lesion.{mask_ext}")

    # Create a mapping of image names to mask paths (strip _lesion from mask.stem)
    image_to_mask = {}
    for mask_path in train_val_test_masks:
        image_name = mask_path.stem.replace('_lesion', '')
        image_to_mask[image_name] = mask_path

    train_val_test_data = []
    unmatched_images = []
    for image_path in train_val_test_images:
        image_name = image_path.stem
        if image_name in image_to_mask:
            train_val_test_data.append((image_path, image_to_mask[image_name]))
        else:
            unmatched_images.append(str(image_path))

    unmatched_masks = []
    image_names = set(img.stem for img in train_val_test_images)
    for mask_path in train_val_test_masks:
        image_name = mask_path.stem.replace('_lesion', '')
        if image_name not in image_names:
            unmatched_masks.append(str(mask_path))

    print(f"Matched {len(train_val_test_data)} image-mask pairs in PH2.")
    if unmatched_images:
        print(f"Images without masks in PH2: {unmatched_images}")
    if unmatched_masks:
        print(f"Masks without images in PH2: {unmatched_masks}")
    if len(train_val_test_data) == 0:
        raise ValueError("No matched image-mask pairs found in PH2.")

    train_count = int(len(train_val_test_data) * train_ratio)
    val_count = int(len(train_val_test_data) * val_ratio)

    if split_random:
        random.shuffle(train_val_test_data)

    train_data = train_val_test_data[:train_count]
    val_data = train_val_test_data[train_count:train_count + val_count]
    test_data = train_val_test_data[train_count + val_count:]

    __save_output(output_dir, train_data, val_data, test_data)


def __save_output(output_dir, train_data, val_data, test_data):
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if len(train_data) != 0:
        fp_images = open(output_dir.joinpath("train_images.txt"), 'w')
        fp_masks = open(output_dir.joinpath("train_masks.txt"), 'w')

        for item in train_data:
            fp_images.write("%s\n" % item[0])
            fp_masks.write("%s\n" % item[1])

        fp_images.close()
        fp_masks.close()

    if len(val_data) != 0:
        fp_images = open(output_dir.joinpath("val_images.txt"), 'w')
        fp_masks = open(output_dir.joinpath("val_masks.txt"), 'w')

        for item in val_data:
            fp_images.write("%s\n" % item[0])
            fp_masks.write("%s\n" % item[1])

        fp_images.close()
        fp_masks.close()

    if len(test_data) != 0:
        fp_images = open(output_dir.joinpath("test_images.txt"), 'w')
        fp_masks = open(output_dir.joinpath("test_masks.txt"), 'w')

        for item in test_data:
            fp_images.write("%s\n" % item[0])
            fp_masks.write("%s\n" % item[1])

        fp_images.close()
        fp_masks.close()


if __name__ == '__main__':
    random_seed = 1234

    isic2016(dataset_dir="datasets/ISIC2016/", output_dir="data/isic2016/",
             random_seed=random_seed)
    isic2016_224x224(dataset_dir="datasets/ISIC2016_224x224/", output_dir="data/isic2016_224x224/", image_ext='png',
             random_seed=random_seed)

    isic2017(dataset_dir="datasets/ISIC2017/", output_dir="data/isic2017/")
    isic2017(dataset_dir="datasets/ISIC2017_224x224/", output_dir="data/isic2017_224x224/", image_ext='png')

    isic2018(dataset_dir="datasets/ISIC2018/", output_dir="data/isic2018/",
             random_seed=random_seed)
    isic2018(dataset_dir="datasets/ISIC2018_224x224/", output_dir="data/isic2018_224x224/", image_ext='png',
             random_seed=random_seed)

    ph2(dataset_dir="datasets/PH2/", output_dir="data/ph2/",
        random_seed=random_seed)
