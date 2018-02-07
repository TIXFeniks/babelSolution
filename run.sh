#!/bin/bash

DATA_PATH="/home/anton/deephack/onsight/en-fr-small"
INPUT_DATA_PATH="/data"
OUTPUT_DATA_PATH="/output"
PROJECT_DIR="/home/anton/deephack/onsight/git"

# Let's keep here pathes for local testing and comment them out
# PROJECT_DIR="."
# DATA_PATH="data"
# INPUT_DATA_PATH="data"
# OUTPUT_DATA_PATH="data"

MODEL_NAME="transformer"
HP_FILE_PATH="$PROJECT_DIR/hp_files/trans_tiny.json"
BATCH_SIZE_FOR_INFERENCE=2
SHOULD_VALIDATE_EVERY_EPOCH=True
MAX_TIME_SECONDS=120
MAX_EPOCHS=1000
USE_EARLY_STOPPING=True
EARLY_STOPPING_LAST_N=500
WARM_UP_NUM_EPOCHS=5

# What the hack is this?
cd "$PROJECT_DIR"


# Training the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --max_time_seconds="$MAX_TIME_SECONDS" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --use_early_stopping="$USE_EARLY_STOPPING" \
            --early_stopping_last_n="$EARLY_STOPPING_LAST_N" \
            --max_epochs="$MAX_EPOCHS" \
            --validate_every_epoch="$SHOULD_VALIDATE_EVERY_EPOCH" \
            --warm_up_num_epochs="$WARM_UP_NUM_EPOCHS"

# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/run.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --model_path="$PROJECT_DIR/trained_models/$MODEL_NAME/model.npz" \
            --input_path="$DATA_PATH/bpe_test_en.txt" \
            --output_path="$OUTPUT_DATA_PATH/output.txt" \
            --hp_file_path="$HP_FILE_PATH" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE"
