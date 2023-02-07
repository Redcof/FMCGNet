export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH="$PYTHONPATH:$(pwd)/customdataset"
export PYTHONPATH="$PYTHONPATH:$(pwd)/model"

export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" # mac
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages"                             # wsl

python model/train_lenet.py \
  --name exp21_0001_FL_defmdila_wv_ep60_iou02 \
  \
  --phase train --niter 60 \
  --dataroot $DATA_ROOT \
  --outf=./output/DET --detection\
  --atz_mypatch \
  \
  --dataset atz \
  --device gpu \
  --atz_patch_db customdataset/atz/atz_patch_dataset__3_128_27_v3_10%_30_99%_multiple_refbox.csv \
  --area_threshold 0.05 \
  --atz_classes "['KK', 'CK', 'CL', 'MD', 'SS', 'GA']" \
  --atz_wavelet "{'wavelet':'sym4', 'method':'VisuShrink','level':3, 'mode':'hard'}" --atz_wavelet_denoise\
  --manualseed 47 \
  --lr 0.0001 \
  --loss FL \
  --iou 0.2 \
   --dilation 2 --deformable
