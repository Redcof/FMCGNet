from patchify import patchify
import pathlib
import os
import cv2
import matplotlib.pyplot as plt

from preprocessing import read_content

root = pathlib.Path("/Users/soumen/Downloads/Datasets/Active Terahertz Imaging Dataset")
image_root = root / "THZ_dataset_det_VOC/JPEGImages"
voc_root = root / "THZ_dataset_det_VOC/Annotations"

# select image and annotation
image = "T_P_M6_LW_V_LL_CL_V_LA_SS_V_B_back_0906154716.jpg"
voc = voc_root / image.replace(".jpg", ".xml")

# read annotation
name, boxes = read_content(str(voc))
# read image
img = cv2.imread(str(image_root / image))
# apply bbox
r, g, b = 255, 0, 0
for box_info in boxes:
    (xmin, ymin, xmax, ymax, cx, cy, class_) = box_info
    x, y, w, h = xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)
    if class_ == "HUMAN":
        continue
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (b, g, r), 4)
    print("boxes:", class_)

# render image
cv2.imshow(image, img)
# create patches
patches = patchify(img, (64, 64, 3), step=32)
print(patches.shape, img.shape)

rows = patches.shape[0]
cols = patches.shape[1]
plt.axis("off")
for r in range(0, rows):
    for c in range(0, cols):
        idx = (r * cols + c + 1)
        im = patches[r, c, 0, :, :]
        ax = plt.subplot(rows, cols, idx)
        ax.axis("off")
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        # cv2.imshow("%d"%idx, im)
plt.show()
cv2.destroyAllWindows()
