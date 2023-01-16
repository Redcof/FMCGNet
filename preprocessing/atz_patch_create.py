import platform
from collections import defaultdict

import numpy as np
import pandas as pd
from empatches import EMPatches
import pathlib
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import shapely.geometry
from PIL import Image

from preprocessing.utils import read_vocxml_content, parsing_filename

class_name_dict = dict(KK="Kitchen Knife", GA="Gun", MD="Metal Dagger",
                       SS="Scissors", WB="Water Bottle", CK="Ceramic Knife",
                       CP="Cell Phone", KC="Key Chain", LW="Leather Wallet",
                       CL="Cigarette Lighter", UN="Unknown", UNKNOWN="UNKNOWN",
                       HUMAN="HUMAN",
                       )
# do not change this order
atz_classes = ["NORMAL0", "NORMAL1", "HUMAN",
               "UNKNOWN", "UN",
               "CP", "MD", "GA", "KK", "SS",
               "KC", "WB", "LW", "CK", "CL"]
# index for a class from the `atz_classes` list. Any class upto 'SS'
# in that list will be ignored during dataset creation, set -1 to consider all classes
atz_ignore_cls_idx_lim = atz_classes.index("HUMAN")

if platform.system() == "Darwin":
    root = pathlib.Path("/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset")
elif platform.system() == "Windows":
    root = pathlib.Path(r"C:\Users\dndlssardar\Downloads")

image_root = root / "THZ_dataset_det_VOC/JPEGImages"
voc_root = root / "THZ_dataset_det_VOC/Annotations"
display = False  # display plots
dataset_save_path = pathlib.Path("../customdataset/atz")
patch_size = 128  # 64, 128

# create save directory
patch_image_save_path = dataset_save_path / "images"
os.makedirs(str(patch_image_save_path), exist_ok=True)

# inter
PATCH_AREA = patch_size ** 2

rejected = defaultdict(lambda: 0)
accepted = defaultdict(lambda: 0)


def is_enough_contrast(rgb, is_anomaly, threshold, percent):
    """
    Inspired by: https://stackoverflow.com/a/596243/4654847
    """
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    perceived_lum = (0.299 * r + 0.587 * g + 0.114 * b)
    cv2.imshow("brightness", perceived_lum)
    perceived_lum2 = np.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)
    cv2.imshow("brightness2", perceived_lum2)
    brightness = np.percentile(perceived_lum, percent)
    if brightness >= threshold:
        return True
    else:
        return False


def is_mostly_black(image, is_anomaly, threshold, percent):
    color = "cyan" if is_anomaly else "red"
    is_anomaly = "anomaly" if is_anomaly else "normal"
    pilimage = Image.fromarray(image.copy()).convert('L')
    # pilimage.show()
    pixels = pilimage.getdata()
    dark_pixel_count = 0
    for pixel in pixels:
        if pixel < threshold:
            dark_pixel_count += 1
    if dark_pixel_count / PATCH_AREA > percent:
        if is_anomaly == "anomaly":
            # ax = plt.subplot(121)
            # ax.set_title("'%s' 3CH image [x]. %d/%d" % (is_anomaly, dark_pixel_count, PATCH_AREA),
            #              fontdict=dict(color=color))
            # plt.imshow(image)
            # ax = plt.subplot(122)
            # ax.set_title("1CH image accepted. %5.4f%%" % (dark_pixel_count / PATCH_AREA))
            # plt.imshow(pilimage, cmap="gray")
            # plt.show()
            rejected['anomaly'] += 1
        else:
            rejected['normal'] += 1
        return True
    else:
        if is_anomaly == "anomaly":
            accepted['anomaly'] += 1
        else:
            accepted['normal'] += 1
        # ax = plt.subplot(121)
        # ax.set_title("'%s' 3CH image [ok]. %d/%d" % (is_anomaly, dark_pixel_count, PATCH_AREA))
        # plt.imshow(image)
        # ax = plt.subplot(122)
        # ax.set_title("1CH image. %5.4f%%" % (dark_pixel_count / PATCH_AREA))
        # plt.imshow(pilimage, cmap="gray")
        # plt.show()
        return False


def good_enough(img_p, is_anomaly):
    """
    If all pixels are black or contains salt-n-pepper noise
    we can return False.
    """
    is_black_image = is_mostly_black(img_p, is_anomaly, threshold=30, percent=0.99)
    return not is_black_image


def rectangle(x1, y1, x2, y2):
    return shapely.geometry.Polygon((
        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    ))


def box_intersect(main_box_x1y1x2y2, subject_box_x1y1x2y2):
    main_rect = rectangle(*main_box_x1y1x2y2)
    subject_rect = rectangle(*subject_box_x1y1x2y2)
    is_intersect = main_rect.intersects(subject_rect)
    if is_intersect:
        intersection = main_rect.intersection(subject_rect)
        intersection_area = intersection.area
        if intersection_area / PATCH_AREA >= 0.1:
            return True
    return False


def create_patch_dataset():
    # select image and annotation
    image_files = os.listdir(image_root)
    atz_patch_dataset_df = pd.DataFrame()
    for image_name in tqdm(image_files):
        if not any([c in image_name for c in atz_classes[atz_ignore_cls_idx_lim + 1:]]):
            continue
        # image_name = 'S_N_M2_LW_L_LA_back_0904130340.jpg'
        voc_xml = voc_root / image_name.replace(".jpg", ".xml")
        base, data = parsing_filename(str(voc_root), image_name.replace(".jpg", ".xml"))

        subject_gender = base['subject_gender']
        subject_id = base['subject_id']
        threat_present = base['presence']
        front_back = base['front_back']

        # read annotation
        name, boxes = read_vocxml_content(str(voc_xml))
        # read image
        img = cv2.imread(str(image_root / image_name))
        # print(img.shape)
        # create a mask to store class info
        mask = np.zeros(img.shape)
        r, g, b = 255, 0, 0
        box_dict = dict()
        # apply bbox and create mask
        for box_info in boxes:
            xmin, ymin, xmax, ymax, cx, cy, label_txt = box_info
            cls_idx = atz_classes.index(label_txt)
            box_dict[label_txt] = xmin, ymin, xmax, ymax
            if cls_idx <= atz_ignore_cls_idx_lim or label_txt == "HUMAN":
                # ignore classes
                continue
            mask[ymin:ymax + 1, xmin:xmax + 1] = cls_idx
            if display:
                x, y, w, h = xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (r, g, b), 4)
                print("boxes:", label_txt, atz_classes.index(label_txt))

        if display:
            # render image
            cv2.imshow("image", img)
            cv2.imshow("mask", mask)
            print(np.unique(mask))

        emp = EMPatches()
        overlap = 0.2
        img_patches, indices = emp.extract_patches(img, patchsize=patch_size, overlap=overlap)
        mask_patches, _ = emp.extract_patches(mask, patchsize=patch_size, overlap=overlap)
        # print(np.unique(mask))
        cols = 4
        rows = len(img_patches) // cols + 1
        dictionary_ls = []
        for idx, (img_p, mask_p, patch_loc) in enumerate(zip(img_patches, mask_patches, indices)):
            max_val = int(np.max(mask_p))
            img_p = img_p.copy()
            mask_p = mask_p.copy()
            # class label
            class_index = max_val

            # class name
            label_txt = atz_classes[class_index]
            if max_val == 0:
                # it's a normal image
                if not box_intersect(box_dict["HUMAN"], patch_loc):
                    continue
            if not good_enough(img_p, class_index > 3):
                continue

            # if class_index >= 3:
            #     plt.imshow(img_p)
            #     plt.title(label_txt)
            #     plt.show()
            global_box = box_dict.get(label_txt, (0, 0, 0, 0))
            mask_p[mask_p > 0] = 1  # replace all non zeros with 1
            # object area in musk
            obj_area_px = np.sum(mask_p.flatten())

            dictionary = dict(image=image_name,
                              threat_present=threat_present,
                              front_back=front_back,
                              patch_id=idx,
                              label=class_index,
                              label_txt=label_txt,
                              global_x1y1x2y2=global_box,
                              anomaly_size=obj_area_px, x1x2y1y2=patch_loc,
                              subject_gender=subject_gender,
                              subject_id="%s%s" % (subject_gender, subject_id))
            # prepare record
            dictionary_ls.append(dictionary)
            # save image
            # cv2.imwrite(str(patch_image_save_path / patch_file_name), img_p)
            if display:
                ax = plt.subplot(rows, cols, idx + 1)
                ax.axis("off")
                mask_rgb = mask_p.copy().astype('uint8')
                mask_rgb[mask_rgb > 0] = 255
                ax.set_title("%d" % idx, fontdict=dict(color="red"))
                ax.imshow(np.hstack((img_p, mask_rgb)))

                # ax = plt.subplot(rows, cols, idx + 2)
                # ax.axis("off")
                # ax.imshow(mask_p)
        # update dataframe
        df_dictionary = pd.DataFrame(dictionary_ls)
        atz_patch_dataset_df = pd.concat([atz_patch_dataset_df, df_dictionary], ignore_index=True)
    # save csv
    postfix = "_%d_%d_%d" % (atz_ignore_cls_idx_lim + 1, patch_size, len(indices))
    patch_dataset_csv = str(dataset_save_path / ("atz_patch_dataset_%s.csv" % postfix))
    atz_patch_dataset_df.columns = dictionary.keys()
    atz_patch_dataset_df.to_csv(patch_dataset_csv)
    # print("Files are saved @", str(patch_image_save_path))
    print("Metadata @", patch_dataset_csv, len(atz_patch_dataset_df), "items")
    print("accepted", accepted)
    print("rejected", rejected)


if __name__ == '__main__':
    for i in [128]:
        patch_size = i
        create_patch_dataset()
    if display:
        plt.show()
        cv2.destroyAllWindows()
