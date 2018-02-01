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

MODEL_NAME="lm"
HP_FILE_PATH="$PROJECT_DIR/hp_files/lm_fitted.json"
MAX_TIME_SECONDS=20
MAX_EPOCHS=5
LANG=$1

# What the hack is this?
cd "$PROJECT_DIR"


# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --lang="$LANG" \
            --max_epochs=$MAX_EPOCHS \
            --max_time_seconds=$MAX_TIME_SECONDS
