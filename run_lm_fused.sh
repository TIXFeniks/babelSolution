#!/bin/bash

DATA_PATH="/nmt/data"
INPUT_DATA_PATH="/data"
OUTPUT_DATA_PATH="/output"
PROJECT_DIR="/nmt"

HP_FILE_PATH="$PROJECT_DIR/hp_files/trans_0_9.json"

# Let's keep here pathes for local testing and comment them out
# PROJECT_DIR="."
# DATA_PATH="data"
# INPUT_DATA_PATH="data_small"
# OUTPUT_DATA_PATH="outputs"

HP_FILE_PATH="$PROJECT_DIR/hp_files/mini_transformer.json"

mosesdecoder=$PROJECT_DIR/ext_libs/mosesdecoder

# Preparing data
$PROJECT_DIR/tokenize.sh "$PROJECT_DIR" "$INPUT_DATA_PATH" 16000 4000

###
# Running first LM model (for source lang)
###

LANG=1
MODEL_NAME="lm$LANG"
MAX_TIME_SECONDS=300
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

LANG=2
MODEL_NAME="lm$LANG"
MAX_TIME_SECONDS=300
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
BATCH_SIZE_FOR_INFERENCE=16
MAX_TIME_SECONDS=300
VALIDATE_EVERY_EPOCH=1 # Validating every epoch
MAX_EPOCHS=1000
USE_EARLY_STOPPING=True
EARLY_STOPPING_LAST_N=10
MAX_NUM_MODELS=4
MIN_INTERVAL_BETWEEN_SAVES=5

PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_fused.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --hp_file_path="$HP_FILE_PATH" \
            --max_time_seconds="$MAX_TIME_SECONDS" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --use_early_stopping="$USE_EARLY_STOPPING" \
            --early_stopping_last_n="$EARLY_STOPPING_LAST_N" \
            --max_epochs="$MAX_EPOCHS" \
            --validate_every_epoch="$VALIDATE_EVERY_EPOCH" \
            --target_lm_path="$PROJECT_DIR/trained_models/lm2/model.npz" \
            --src_lm_path="$PROJECT_DIR/trained_models/lm1/model.npz" \
            --max_num_models="$MAX_NUM_MODELS" \
            --min_interval_between_saves="$MIN_INTERVAL_BETWEEN_SAVES"

# Running the model
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/run_fused.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --models_dir="$PROJECT_DIR/trained_models/$MODEL_NAME" \
            --input_path="$DATA_PATH/bpe_input.txt" \
            --output_path="$DATA_PATH/output.tok.txt" \
            --hp_file_path="$HP_FILE_PATH" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --target_lm_path="$PROJECT_DIR/trained_models/lm2/model.npz"


cat $DATA_PATH/output.tok.txt | $mosesdecoder/scripts/tokenizer/detokenizer.perl > $OUTPUT_DATA_PATH/output.txt

python3.6 final_fix.py $DATA_PATH/output.tok.txt $OUTPUT_DATA_PATH/output.txt
