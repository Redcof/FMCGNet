export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH="$PYTHONPATH:$(pwd)/customdataset"
export PYTHONPATH="$PYTHONPATH:$(pwd)/model"

export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" # mac
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages"                             # wsl

python model/train_lenet.py \
  --name exp9_modlenet_deform_dil2__wavelet_focal_loss_lr_009 \
  \
  --phase train --niter 50 \
  --dataroot $DATA_ROOT \
  \
  --dataset atz \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_36_v2_10%_30_99%.csv \
  --area_threshold 0.05 \
  --atz_classes "['KK', 'CK', 'CL', 'MD', 'SS', 'GA']" \
  --atz_wavelet "{'wavelet':'sym4', 'method':'VisuShrink','level':3, 'mode':'hard'}" \
  --manualseed 47 --atz_wavelet_denoise --dilation 2 --loss FL --lr 0.009 --deformable
