"""
LOAD DATA from file.
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
import ast
import os
import pathlib

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

from customdataset.atz.atzdetdataset import ATZDetDataset
from preprocessing.image_transform import wavelet_denoise_rgb

NORMAL_CLASSES = ["NORMAL0", "NORMAL1"]
CLASSE_IDX = ["UNKNOWN", "CP", "MD", "GA", "KK", "SS", "KC", "WB", "LW", "CK", "CL"]


def label_transform(num_classes, PATCH_AREA, object_area_threshold):
    """ This label transform is designed for SAGAN.
    Return 0 for normal images and 1 for abnormal images """

    def internal(image, label, anomaly_size_px):
        normal = 0
        anomaly = 1
        # object area in patch must be bigger than some threshold
        if anomaly_size_px > 0:
            object_area_percent = anomaly_size_px / PATCH_AREA
        else:
            object_area_percent = 0
        onehot_labels = np.zeros((1, num_classes))
        if (label in NORMAL_CLASSES
                or anomaly_size_px == 0
                # not in iou range
                or not (1 >= object_area_percent >= object_area_threshold)):
            onehot_labels[0, 0] = 1
            return onehot_labels, normal
        else:
            idx = CLASSE_IDX.index(label)
            onehot_labels[0, idx] = 1
            return onehot_labels, idx

    return internal


def wavelet_transform(atz_wavelet):
    def internal(x):
        is_converted = False
        if isinstance(x, Image.Image):
            # if PIL convert to numpy
            is_converted = True
            x = np.array(x)

        x = wavelet_denoise_rgb(x,
                                channel_axis=2,
                                wavelet=atz_wavelet['wavelet'],  # 'bior6.8'
                                method=atz_wavelet['method'],  # 'VisuShrink',
                                decomposition_level=atz_wavelet['level'],  # 1
                                threshold_mode=atz_wavelet['mode']  # 'hard'
                                )

        if is_converted:
            # restore back to PIL if converted previously
            x = Image.fromarray(x)
        return x

    return internal


##
def load_atz_data(opt, transform=None):
    """ Load Data

    Args:
        opt ([type]): Argument Parser
        transform: Torch transforms

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    assert os.path.exists(opt.atz_patch_db)  # mandatory for ATZ

    curr = pathlib.Path(__file__).parents[0]

    patch_dataset_csv = opt.atz_patch_db
    atz_ablation = opt.atz_ablation
    torch_device = torch.device("cuda:0" if opt.device != 'cpu' else "cpu")

    try:
        atz_classes = ast.literal_eval(opt.atz_classes)
    except ValueError:
        atz_classes = []
    atz_classes.extend(NORMAL_CLASSES)

    try:
        atz_subjects = ast.literal_eval(opt.atz_subjects)
    except ValueError:
        atz_subjects = []

    try:
        atz_wavelet = ast.literal_eval(opt.atz_wavelet)
    except ValueError:
        atz_wavelet = {'wavelet': 'sym4', 'method': 'VisuShrink', 'level': 1, 'mode': 'soft'}

    object_area_threshold = opt.area_threshold  # 10%
    patchsize = opt.isize
    PATCH_AREA = patchsize ** 2

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((opt.isize, opt.isize)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    num_classes = len(CLASSE_IDX)

    train_ds = ATZDetDataset(patch_dataset_csv, opt.dataroot, "test",
                             object_only=opt.detection,
                             detection=opt.detection,
                             empatch=not opt.atz_mypatch,
                             atz_dataset_train_or_test_txt=str(curr / 'train.txt'),
                             device=torch_device,
                             classes=atz_classes,
                             patch_size=opt.isize,
                             patch_overlap=opt.atz_patch_overlap,
                             subjects=atz_subjects,
                             ablation=atz_ablation,
                             balanced=True,
                             nc=opt.nc,
                             transform=transform,
                             random_state=opt.manualseed,
                             label_transform=label_transform(num_classes, PATCH_AREA, object_area_threshold),
                             global_wavelet_transform=wavelet_transform(
                                 atz_wavelet) if opt.atz_wavelet_denoise else None)

    test_ds = ATZDetDataset(patch_dataset_csv, opt.dataroot, "test",
                            object_only=opt.detection,
                            detection=opt.detection,
                            empatch=not opt.atz_mypatch,
                            atz_dataset_train_or_test_txt=str(curr / 'test.txt'),
                            device=torch_device,
                            classes=atz_classes,
                            patch_size=opt.isize,
                            patch_overlap=opt.atz_patch_overlap,
                            subjects=atz_subjects,
                            ablation=atz_ablation,
                            balanced=True,
                            nc=opt.nc,
                            transform=transform,
                            random_state=opt.manualseed,
                            label_transform=label_transform(num_classes, PATCH_AREA, object_area_threshold),
                            global_wavelet_transform=wavelet_transform(
                                atz_wavelet) if opt.atz_wavelet_denoise else None)

    valid_ds = ATZDetDataset(patch_dataset_csv, opt.dataroot, "test",
                             object_only=opt.detection,
                             detection=opt.detection,
                             empatch=not opt.atz_mypatch,
                             atz_dataset_train_or_test_txt=str(curr / 'val.txt'),
                             device=torch_device,
                             classes=atz_classes,
                             patch_size=opt.isize,
                             patch_overlap=opt.atz_patch_overlap,
                             subjects=atz_subjects,
                             ablation=atz_ablation,
                             balanced=True,
                             nc=opt.nc,
                             transform=transform,
                             random_state=opt.manualseed,
                             label_transform=label_transform(num_classes, PATCH_AREA, object_area_threshold),
                             global_wavelet_transform=wavelet_transform(
                                 atz_wavelet) if opt.atz_wavelet_denoise else None)
    opt.log("Train Dataset '%s' => Normal:Abnormal = %d:%d" % ("train", train_ds.normal_count, train_ds.abnormal_count))
    opt.log("Test Dataset '%s' => Normal:Abnormal = %d:%d" % ("test", test_ds.normal_count, test_ds.abnormal_count))
    opt.log(
        "Validation Dataset '%s' => Normal:Abnormal = %d:%d" % ("val", valid_ds.normal_count, valid_ds.abnormal_count))

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=False, drop_last=True)
    test_dl = DataLoader(dataset=test_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return train_dl, test_dl, valid_dl
