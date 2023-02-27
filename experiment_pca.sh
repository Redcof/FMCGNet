export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONPATH="$PYTHONPATH:$(pwd)/customdataset"
export PYTHONPATH="$PYTHONPATH:$(pwd)/model"

export DATA_ROOT="/Users/soumen/Downloads/Datasets/ActiveTerahertzImagingDataset/THZ_dataset_det_VOC/JPEGImages" # mac
export DATA_ROOT="/mnt/c/Users/dndlssardar/Downloads/THZ_dataset_det_VOC/JPEGImages"                             # wsl

python preprocessing/dimensionality_reduction.py \
--phase train \
--batchsize 64 --device cpu \
--dataroot $DATA_ROOT \
--dataset atz \
--atz_patch_db customdataset/atz/atz_patch_dataset__3_128_27_v3_10%_30_99%_multiple_refbox.csv --atz_mypatch \
--area_threshold 0.05 \
--nc 1 \
--atz_classes "['KK', 'CK', 'CL', 'MD', 'SS', 'GA']" \
--atz_wavelet "{'wavelet':'sym4', 'method':'VisuShrink','level':2, 'mode':'hard'}" \
--manualseed 47