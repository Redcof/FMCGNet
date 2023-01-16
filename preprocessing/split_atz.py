import shutil
import random
from patchify import patchify
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from atz_preprocess_options import ATZPreprocessOptions
from preprocessing.utils import read_vocxml_content


def copy_files(files, src_root, dest_root, save_image, txt_file):
    with open(txt_file, "a") as fp:
        for im_name in tqdm(files):
            # objects are present
            im_src = str(src_root / im_name)
            im_dest = str(dest_root / im_name)
            if save_image:
                shutil.copy(im_src, im_dest)
            fp.write("%s\n" % im_name)


def create_dataset(src_dir, dataset_dir, is_normal_func, opt):
    ablation = opt.ablation
    train_split = opt.train_split
    save_image = opt.save_image
    filelist_path = opt.filelist
    train_normal_path = dataset_dir / "train" / "0.normal"
    test_normal_path = dataset_dir / "test" / "0.normal"
    test_abnormal_path = dataset_dir / "test" / "1.abnormal"
    train_dataset_txt = str(dataset_dir / "atz_dataset_train_postfix.txt")
    test_dataset_txt = str(dataset_dir / "atz_dataset_test_postfix.txt")

    os.makedirs(str(dataset_dir), exist_ok=True)
    # cleanup
    shutil.rmtree(str(dataset_dir / "train"), ignore_errors=True)
    shutil.rmtree(str(dataset_dir / "test"), ignore_errors=True)

    if save_image:
        # create folder if required
        os.makedirs(str(train_normal_path))
        os.makedirs(str(test_normal_path))
        os.makedirs(str(test_abnormal_path))

    # list images
    if filelist_path is None:
        files = os.listdir(str(src_dir))
    else:
        with open(filelist_path, "r") as fp:
            files = [fname.strip() for fname in fp.readlines()]

    # random shuffle
    random.seed(47)  # setting seed for reproducibility
    random.shuffle(files)

    # select image and annotation
    normal = []
    abnormal = []

    # separate files into normal and abnormal
    for image_name in files:
        if is_normal_func(image_name):
            normal.append(image_name)
        else:
            abnormal.append(image_name)

    # shuffle filenames
    random.seed(47)  # setting seed for reproducibility
    random.shuffle(normal)
    random.seed(47)  # setting seed for reproducibility
    random.shuffle(abnormal)

    if ablation > 0:
        normal = normal[:ablation * 2]
        abnormal = abnormal[:ablation]
        train_dataset_txt = train_dataset_txt.replace("postfix", ("ablation_%d" % ablation))
        test_dataset_txt = test_dataset_txt.replace("postfix", ("ablation_%d" % ablation))
        print("Preparing data for ablation study.")
    else:
        train_dataset_txt = train_dataset_txt.replace("postfix", "")
        test_dataset_txt = test_dataset_txt.replace("postfix", "")
        print("Preparing full data.")

    # cleanup dataset files
    try:
        os.remove(train_dataset_txt)
    except FileNotFoundError:
        ...
    try:
        os.remove(test_dataset_txt)
    except FileNotFoundError:
        ...

    # split normal into train and test
    _80 = int(len(normal) * train_split)
    train_split = normal[:_80]
    test_split = normal[_80:]

    # copy abnormal files
    copy_files(abnormal, src_dir, test_abnormal_path, save_image, test_dataset_txt)

    # copy normal train files
    copy_files(train_split, src_dir, train_normal_path, save_image, train_dataset_txt)

    # copy normal test files
    copy_files(test_split, src_dir, test_normal_path, save_image, test_dataset_txt)
    if save_image:
        print("Dataset saved @", str(dataset_dir))
    print("Train dataset created @", str(train_dataset_txt))
    print("Test dataset created @", str(test_dataset_txt))


# create a dataset for SAGAN from ATZ
img_root = pathlib.Path("/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset")
voc_root = img_root / "THZ_dataset_det_VOC/Annotations"


def check_normal(path):
    global voc_root
    voc = voc_root / path.replace(".jpg", ".xml")
    name, boxes = read_vocxml_content(voc)
    return len(boxes) == 1


if __name__ == '__main__':
    opt = ATZPreprocessOptions().parse()
    src = img_root / "THZ_dataset_det_VOC/JPEGImages"
    atz_sagan_data_dir = pathlib.Path(opt.save_path)
    create_dataset(src, atz_sagan_data_dir, check_normal, opt)
