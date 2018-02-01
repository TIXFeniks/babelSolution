#!/bin/bash

DATA_PATH="/nmt/data"
INPUT_DATA_PATH="/data"
OUTPUT_DATA_PATH="/output"
PROJECT_DIR="/nmt"

# Let's keep here pathes for local testing and comment them out
PROJECT_DIR="."
DATA_PATH="data"
INPUT_DATA_PATH="data"
OUTPUT_DATA_PATH="data"

MODEL_NAME="transformer"
HP_FILE_PATH="$PROJECT_DIR/hp_files/lm_fitted.json"
MAX_TIME_SECONDS=180
MAX_EPOCHS=5
USE_EARLY_STOPPING=True
EARLY_STOPPING_LAST_N=5
LANG=2

# What the hack is this?
cd "$PROJECT_DIR"


# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --model_path="$PROJECT_DIR/trained_models/$MODEL_NAME/model.npz" \
            --input_path="$DATA_PATH/bpe_input.txt" \
            --output_path="$OUTPUT_DATA_PATH/output.txt" \
            --hp_file_path="$HP_FILE_PATH" \
            --lang="$LANG"
