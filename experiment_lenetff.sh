export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH="$PYTHONPATH:$(pwd)/customdataset"
export PYTHONPATH="$PYTHONPATH:$(pwd)/model"

export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" # mac
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages"                             # wsl

python model/train_lenet.py \
  --name exp6_fflenet_deform_dil2__wavelet_lr_0001 \
  \
  --phase train --niter 40 \
  --dataroot $DATA_ROOT \
  --device 'gpu' \
  --outf=./output/FF --ff\
  \
  --dataset atz \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36_v2_10%_30_99%.csv \
  --area_threshold 0.05 \
  --atz_classes "['KK', 'CK', 'CL', 'MD', 'SS', 'GA']" \
  --atz_wavelet "{'wavelet':'sym4', 'method':'VisuShrink','level':3, 'mode':'hard'}" \
  --manualseed 47 --dilation 2 --lr 0.0001 --ffgoodness 2 --atz_wavelet_denoise --deformable
