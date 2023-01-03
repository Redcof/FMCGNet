import xml.etree.ElementTree as ET
from io import StringIO, BytesIO


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

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
