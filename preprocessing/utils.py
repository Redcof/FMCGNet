import hashlib
import os
import pathlib
import platform
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO

import numpy as np
import pandas as pd

if platform.system() == "Darwin":
    root_dir = pathlib.Path(r"/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC")
elif platform.system() == "Windows":
    root_dir = pathlib.Path(r"C:\Users\dndlssardar\Downloads\THZ_dataset_det_VOC")


annotation_dir = root_dir / "Annotations"
img_dir = root_dir / "JPEGImages"

train_data = annotation_dir / "train.txt"
test_data = annotation_dir / "test.txt"
val_data = annotation_dir / "val.txt"

class_name_dict = dict(KK="Kitchen Knife", GA="Gun", MD="Metal Dagger",
                       SS="Scissors", WB="Water Bottle", CK="Ceramic Knife",
                       CP="Cell Phone", KC="Key Chain", LW="Leather Wallet",
                       CL="Cigarette Lighter", UN="Unknown", UNKNOWN="UNKNOWN",
                       #
                       # Unspecified="Unspecified",
                       HUMAN="HUMAN",
                       )

orientation_dict = dict(F="Front", V="Back", L="Left", R="Right",
                        #
                        # Unspecified="Unspecified",
                        HUMAN="HUMAN",
                        MISSING="MISSING",
                        )

location_on_body_dict = dict(LA="Left Arm", RA="Right Arm",
                             LT="Left Thai", RT="Right Thai",
                             LL="Left Calf", RL="Right Calf",
                             B="Back", C="Chest", N="Hip", W="Waist", S="Abdomen",
                             #
                             # Unspecified="Unspecified",
                             HUMAN="HUMAN",
                             MISSING="MISSING",
                             )


# a dummy decoder function to decode abbreviated labels
def decode(key):
    decoder_dict = dict(class_name=class_name_dict,
                        location_on_body=location_on_body_dict,
                        item_orientation=orientation_dict,
                        )[key]

    def internal(x):
        try:
            return decoder_dict[x]
        except KeyError as e:
            print(x)
            raise e

    return internal


def read_vocxml_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    filename = root.find('filename').text
    for boxes in root.iter('object'):
        class_ = boxes.find("name").text
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2

        list_with_single_boxes = (xmin, ymin, xmax, ymax, cx, cy, class_)
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


ctr = 1


def parsing_filename(path, xml):
    global ctr

    # 2. N and P indicate whether there is an item: P (there is an item), N (there is no item)
    # 3. S, D, T represent the number of items: S (single item), D (double items), T (three items).
    # Example: T_N_F4_LW_V_LL_MD_V_RL_SS_V_N_back_0910164725.xml   Since the item is marked as 'N',
    # there is only one 'human' object.
    #        D_P_F3_GA_F_RT_KK_F_LT_front_0910151647.xml Since the item is marked as 'P',
    #        so there are three objects (including 'human' + double items).
    # 4. Any image will have a 'human' object, so the minimum number of objects is '1'

    file_id = hashlib.md5(xml.encode()).hexdigest()
    # split into components
    components = xml.replace(".xml", "").split("_")
    # item availability
    if components[1] not in ["P", "N"]:
        raise ValueError("Unrecognised value at position 2: {0}".format(components[1]))
    # item presence
    presence = 1 if "P" == components[1] else 0
    # calculate no.of items
    items = 1 if not presence else (3 + 1 if components[0] == "T" else (
        2 + 1 if components[0] == "D" else (1 + 1 if components[0] == "S" else 0 + 1)))
    # get timestamp
    timestamp = components[-1]
    front_back = components[-2]
    subject_gender = components[2][0]
    subject_id = components[2][1]

    # get xml elements
    xml_file = str(pathlib.Path(path) / xml)
    name, boxes = read_vocxml_content(xml_file)

    ctr += 1
    # parse class class_names
    classes = []
    metadata = dict(HUMAN=dict(item_orientation="HUMAN",
                               location_on_body="HUMAN"))
    for i in range(1, 4):
        class_name = components[3 * i]
        if class_name in ["front", "back"]:
            break
        item_orientation = components[3 * i + 1]
        location_on_body = components[3 * i + 2]
        if class_name not in class_name_dict.keys():
            raise Exception("Label not found %s@%s, %d" % (class_name, xml, 3 * i))
        if item_orientation not in orientation_dict.keys():
            raise Exception("Orientation not found %s@%s, %d" % (item_orientation, xml, 3 * i + 1))
        if location_on_body not in location_on_body_dict.keys():
            raise Exception("Location not found %s@%s, %d" % (location_on_body, xml, 3 * i + 2))
        metadata[class_name] = dict(item_orientation=item_orientation,
                                    location_on_body=location_on_body
                                    )

    for box in boxes:
        xmin, ymin, xmax, ymax, cx, cy, class_ = box
        try:
            metainf = metadata[class_]
        except KeyError as e:
            metainf = dict(item_orientation="MISSING", location_on_body="MISSING")

        classes.append(dict(class_name=class_,
                            item_orientation=metainf["item_orientation"],
                            location_on_body=metainf["location_on_body"],
                            subject_gender=subject_gender,
                            subject_id=subject_id,
                            front_back=front_back,
                            xmin=xmin,
                            ymin=ymin,
                            xmax=xmax,
                            ymax=ymax,
                            cx=cx,
                            cy=cy,
                            file_id=name
                            ))
    keys_filename = [k for k in metadata.keys() if k != "HUMAN"]
    key_xml_content = [class_ for _, _, _, _, _, _, class_ in boxes]
    key_xml_content2 = [c for c in key_xml_content if c != "HUMAN"]
    # prepare primary meta data
    base = dict(gross_count=(items - 1),  # item present excluding HUMAN
                char1=components[0],  # 1st character
                char2=components[1],  # 2nd character
                presence=presence,  # 2nd character status, item availability
                subject_gender=subject_gender,  # subject gender
                subject_id=subject_id,  # subject id
                front_back=front_back,  # font or back of the subject
                time=timestamp,  # timestamp
                mismatch=0 if all([k in keys_filename for k in key_xml_content2]) else 1,
                # missmatch in character1 and xml bbox
                filename_bbox_count=items,  # filename wise item count including HUMAN
                xml_content_bbox_count=len(boxes),  # xml concet wise item count includign HUMAN
                file_id=file_id,  # md5 filename hash
                keys_filename="{0}".format(keys_filename),  # filename wise class names
                key_xml_content="{0}".format(key_xml_content2),  # xml wise class names
                human_key="HUMAN" if "HUMAN" in key_xml_content else 0,  # HUMAN class present or not
                file=name,  # actual file name
                )
    return base, classes

#
# columns1 = ["char1", "char2", "filename_bbox_count", "xml_content_bbox_count", "gross_count", "presence", "mismatch",
#             "keys_filename", "key_xml_content", "human_key",
#             "subject_gender", "subject_id", "front_back", "time", "file", "file_id", ]
#
# columns2 = ["file_id", "item_orientation", "location_on_body", "class_name", "subject_gender", "subject_id",
#             "front_back", "xmin", "ymin", "xmax", "ymax", "cx", "cy"]
#
#


# ==========
# df1 = pd.DataFrame(columns=columns1)
# df2 = pd.DataFrame(columns=columns2)
# # parsing xml names
# for path, dirs, xml_files in os.walk(str(annotation_dir)):
#     for xml in xml_files:
#         meta, data = parsing_filename(path, xml)
#         dft1 = pd.DataFrame([meta], columns=columns1)
#         dft2 = pd.DataFrame(data, columns=columns2)
#         df1 = pd.concat([df1, dft1], ignore_index=True, verify_integrity=False, sort=False)
#         df2 = pd.concat([df2, dft2], ignore_index=True, verify_integrity=False, sort=False)


# # add more data
# df2["location_on_body_decoded"] = df2["location_on_body"].apply(decode("location_on_body"))
# df2["class_name_decoded"] = df2["class_name"].apply(decode("class_name"))
# df2["item_orientation_decoded"] = df2["item_orientation"].apply(decode("item_orientation"))
# df2["area"] = np.abs(df2["xmax"] - df2["xmin"]) * np.abs(df2["ymax"] - df2["ymin"])
# df2["height"] = np.abs(df2["ymax"] - df2["ymin"])
# df2["width"] = np.abs(df2["xmax"] - df2["xmin"])
# df2.sample(5, random_state=1)
