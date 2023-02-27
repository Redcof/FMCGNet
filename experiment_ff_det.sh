export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH="$PYTHONPATH:$(pwd)/customdataset"
export PYTHONPATH="$PYTHONPATH:$(pwd)/model"

export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" # mac
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages"                             # wsl

python model/train_lenet.py \
--name abl25_ffdet__hybrid_lr_0001_ep600_iou05_b64_onlineaugmentation --phase train \
--batchsize 16 --niter 1 --device gpu \
--dataroot $DATA_ROOT --outf=./output/DET --detection --iou 0.5 \
--dataset atz \
--atz_patch_db customdataset/atz/atz_patch_dataset__3_128_27_v3_10%_30_99%_multiple_refbox.csv --atz_mypatch \
--area_threshold 0.05 \
--nc 1 \
--atz_classes "['KK', 'CK', 'CL', 'MD', 'SS', 'GA']" \
--atz_wavelet "{'wavelet':'sym4', 'method':'VisuShrink','level':2, 'mode':'hard'}" \
--manualseed 47 --lr 0.0001 \
--ff --ffnegalg overlay \
# --ffgoodness 2  --batchnorm --layernorm --dropout