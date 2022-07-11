#!/bin/sh

PTH_SCRIPT=prepare_one_scan_pth.py
GT_SCRIPT=prepare_one_scan_gt.py
TRAIN_IDS=meta_data/scannetv2_train.txt
VAL_IDS=meta_data/scannetv2_val.txt
CONFIG_PATH=../../conf/config.yaml
PARALLEL_N=14

echo "===================================="
echo "Processing train split ..."
echo "===================================="

parallel -j $PARALLEL_N --bar "python $PTH_SCRIPT \
    --id {1} \
    --split train \
    --cfg $CONFIG_PATH" :::: $TRAIN_IDS


echo "===================================="
echo "Processing val split ..."
echo "===================================="

parallel -j $PARALLEL_N --bar "python $PTH_SCRIPT \
    --id {1} \
    --split val \
    --cfg $CONFIG_PATH" :::: $VAL_IDS


echo "===================================="
echo "Processing val split groud truth ..."
echo "===================================="

parallel -j $PARALLEL_N --bar "python $GT_SCRIPT \
    --id {1} \
    --split val \
    --cfg $CONFIG_PATH" :::: $VAL_IDS
