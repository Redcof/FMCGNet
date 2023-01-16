import os

import numpy as np
from PIL import Image
import cv2
from empatches import EMPatches

from preprocessing.wavelet_transform import wavelet_denoise_rgb

w = 128


def crop_test(npmain, nppatch, x1x2y1y2):
    x1, x2, y1, y2 = x1x2y1y2
    pil = Image.fromarray(npmain)
    pil = pil.crop(box=(x1, y1, x2, y2))
    pil.show("Patch-PIL")

    cv2.imshow("Patch-np", npmain[y1:y2, x1:x2])
    cv2.imshow("Patch-orig", nppatch)
    deno = wavelet_denoise_rgb(nppatch, 2, wavelet='bior4.4', method='VisuShrink',
                               decomposition_level=2,
                               threshold_mode='soft')
    cv2.imshow("Deno", deno)
    cv2.waitKey(-1)


if __name__ == '__main__':
    root = "/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages"
    file = "T_P_M6_MD_F_LL_CK_F_C_WB_F_RT_front_0906154134.jpg"
    img_bgr = cv2.imread(os.path.join(root, file))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    emp = EMPatches()
    overlap = 0.2
    img_patches, indices = emp.extract_patches(img_rgb.copy(), patchsize=w, overlap=overlap)
    idx = 10
    patch, loc = img_patches[idx], indices[idx]
    crop_test(img_rgb, patch, loc)
