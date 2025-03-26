#!/bin/bash -ex

EXP_NAME=${EXP_NAME:-en-ar-tts}
DATA_DIR=en-ar-tts-data
VOCAB_TXT=${VOCAB_FILE:-/llms1/mshahmmer/tts_training/F5-TTS-2/F5-TTS/data/arabic-ft-2_custom/vocab.txt}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-qcri/multi_gpu.yaml}
ACCELERATE="accelerate launch --config_file $ACCELERATE_CONFIG"

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
if [ ! -f "data/${EXP_NAME}_custom/vocab.txt" ]; then
    cp $VOCAB_TXT data/${EXP_NAME}_custom/vocab.txt
fi


$ACCELERATE src/f5_tts/train/train.py --config-name F5TTS_v1_en_ar.yaml
