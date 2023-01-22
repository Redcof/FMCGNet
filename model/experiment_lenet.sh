export PYTHONPATH="$PYTHONPATH:$(pwd)"

export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" # mac
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages"                             # wsl

python train_lenet.py \
  --phase train--niter 50 \
  --dataroot $DATA_ROOT \
  --name exp1_lenet_default
