#!/bin/sh

PTH_SCRIPT=prepare_one_scan_pth.py
TRAIN_IDS=meta_data/scannetv2_train.txt
VAL_IDS=meta_data/scannetv2_val.txt
CONFIG_PATH=../../config/config.yaml

PROCESSES=4  # default: using 4 processes in parallel

# read arguments
while getopts j: flag
do
    case "${flag}" in
        j) PROCESSES=${OPTARG};;
    esac
done

echo "\nUsing $PROCESSES processes in parallel ...\n"

echo "============================="
echo "Processing train split ..."
echo "============================="

parallel -j $PROCESSES --bar "python $PTH_SCRIPT \
    --id {1} \
    --split train \
    --cfg $CONFIG_PATH" :::: $TRAIN_IDS


echo "============================="
echo "Processing val split ..."
echo "============================="

parallel -j $PROCESSES --bar "python $PTH_SCRIPT \
    --id {1} \
    --split val \
    --cfg $CONFIG_PATH" :::: $VAL_IDS
