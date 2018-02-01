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

# What the hack is this?
cd "$PROJECT_DIR"

# Preparing data
$PROJECT_DIR/tokenize.sh "$PROJECT_DIR" "$INPUT_DATA_PATH"

LANG=1
MODEL_NAME="lm$LANG"
HP_FILE_PATH="$PROJECT_DIR/hp_files/lm_fitted.json"
MAX_TIME_SECONDS=3600
MAX_EPOCHS=3


# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --lang="$LANG" \
            --max_epochs=$MAX_EPOCHS \
            --max_time_seconds=$MAX_TIME_SECONDS

LANG=2
MODEL_NAME="lm$LANG"
HP_FILE_PATH="$PROJECT_DIR/hp_files/lm_fitted.json"
MAX_TIME_SECONDS=3600
MAX_EPOCHS=3

# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --lang="$LANG" \
            --max_epochs=$MAX_EPOCHS \
            --max_time_seconds=$MAX_TIME_SECONDS

MODEL_NAME="transformer"
HP_FILE_PATH="$PROJECT_DIR/hp_files/lm_fitted.json"
BATCH_SIZE_FOR_INFERENCE=32
MAX_TIME_SECONDS=1800
VALIDATE_EVERY=1000
MAX_EPOCHS=50000
USE_EARLY_STOPPING=True
EARLY_STOPPING_LAST_N=5

# Training the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_fused.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --validate_every="$VALIDATE_EVERY" \
            --max_time_seconds="$MAX_TIME_SECONDS" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --use_early_stopping="$USE_EARLY_STOPPING" \
            --early_stopping_last_n="$EARLY_STOPPING_LAST_N" \
            --max_epochs="$MAX_EPOCHS" \
            --target_lm_path="$PROJECT_DIR/trained_models/lm2/model.npz" \
            --src_lm_path="$PROJECT_DIR/trained_models/lm1/model.npz"

# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/run_fused.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --model_path="$PROJECT_DIR/trained_models/$MODEL_NAME/model.npz" \
            --input_path="$DATA_PATH/bpe_input.txt" \
            --output_path="$OUTPUT_DATA_PATH/output.txt" \
            --hp_file_path="$HP_FILE_PATH" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --target_lm_path="$PROJECT_DIR/trained_models/lm2/model.npz"
