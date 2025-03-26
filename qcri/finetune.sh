#!/bin/bash -ex

EXP_NAME="qcri_tts"
DATA_DIR=/llms1/yifan/F5-TTS/qcri-data

# setup conda
source ~/miniconda3/bin/activate
conda activate f5-tts

# prepare data
if [ ! -d "$DATA_DIR" ]; then
    echo "Please use the generate dataset:"
    echo "./src/f5_tts/train/datasets/prepare_csv_wavs.py --workers 64 \\"
    echo "    $DATA_DIR qcri-data"
    exit 1
fi

if [ ! -d "data/${EXP_NAME}_custom" ]; then
    mkdir -p data/${EXP_NAME}_custom
    cp -r $DATA_DIR/* data/${EXP_NAME}_custom
fi

# make vocab.txt
