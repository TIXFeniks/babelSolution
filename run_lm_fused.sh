#!/bin/bash

DATA_PATH="/home/anton/deephack/onsight/en-fr-small"
INPUT_DATA_PATH="/data"
OUTPUT_DATA_PATH="/output"
PROJECT_DIR="/home/anton/deephack/onsight/git"

HP_FILE_PATH="$PROJECT_DIR/hp_files/trans_tiny.json"

# Let's keep here pathes for local testing and comment them out
# PROJECT_DIR="."
# DATA_PATH="data"
# INPUT_DATA_PATH="data"
# OUTPUT_DATA_PATH="data"

# What the hack is this?
cd "$PROJECT_DIR"

# Preparing data
#$PROJECT_DIR/tokenize.sh "$PROJECT_DIR" "$INPUT_DATA_PATH" 16000 4000

###
# Running first LM model (for source lang)
###

LANG=en
MODEL_NAME="lm$LANG"
MAX_TIME_SECONDS=3600
MAX_EPOCHS=100

# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --lang="$LANG" \
            --max_epochs=$MAX_EPOCHS \
            --max_time_seconds=$MAX_TIME_SECONDS

###
# Running second LM model (for target lang)
###

LANG=fr
MODEL_NAME="lm$LANG"
MAX_TIME_SECONDS=3600
MAX_EPOCHS=100

# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --lang="$LANG" \
            --max_epochs=$MAX_EPOCHS \
            --max_time_seconds=$MAX_TIME_SECONDS


###########
# Running transformer with fused LM
###########

MODEL_NAME="transformer"
BATCH_SIZE_FOR_INFERENCE=2
MAX_TIME_SECONDS=14400
SHOULD_VALIDATE_EVERY_EPOCH=True
MAX_EPOCHS=1000
USE_EARLY_STOPPING=True
EARLY_STOPPING_LAST_N=30
WARM_UP_NUM_EPOCHS=25

# Training the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_fused.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --max_time_seconds="$MAX_TIME_SECONDS" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --use_early_stopping="$USE_EARLY_STOPPING" \
            --early_stopping_last_n="$EARLY_STOPPING_LAST_N" \
            --max_epochs="$MAX_EPOCHS" \
            --validate_every_epoch="$SHOULD_VALIDATE_EVERY_EPOCH" \
            --target_lm_path="$PROJECT_DIR/trained_models/lmfr/model.npz" \
            --src_lm_path="$PROJECT_DIR/trained_models/lmen/model.npz" \
            --warm_up_num_epochs="$WARM_UP_NUM_EPOCHS"

# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/run_fused.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --model_path="$PROJECT_DIR/trained_models/$MODEL_NAME/model.npz" \
            --input_path="$DATA_PATH/bpe_test_en.txt" \
            --output_path="$DATA_PATH/output.tok.txt" \
            --hp_file_path="$HP_FILE_PATH" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --target_lm_path="$PROJECT_DIR/trained_models/lmfr/model.npz"


cat $DATA_PATH/output.tok.txt  > $OUTPUT_DATA_PATH/output.txt

#.6 final_fix.py $DATA_PATH/output.tok.txt $OUTPUT_DATA_PATH/output.txt
