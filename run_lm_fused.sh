#!/bin/bash

DATA_PATH="$1"
INPUT_DATA_PATH="/data"
OUTPUT_DATA_PATH="/output"
PROJECT_DIR="./"

if [ -z "$2" ]; then
    TRAIN_LM=true
else
    TRAIN_LM=$2
fi

if [ -z "$3" ]; then
    TRAIN_TR=true
else
    TRAIN_TR=$3
fi

HP_FILE_PATH="$PROJECT_DIR/hp_files/trans_default.json"



# Let's keep here pathes for local testing and comment them out
# PROJECT_DIR="."
# DATA_PATH="data"
# INPUT_DATA_PATH="data"
# OUTPUT_DATA_PATH="data"

#mosesdecoder=$PROJECT_DIR/ext_libs/mosesdecoder

# What the hack is this?
cd "$PROJECT_DIR"

# Preparing data
#$PROJECT_DIR/tokenize.sh "$PROJECT_DIR" "$INPUT_DATA_PATH" 16000 4000

###
# Running first LM model (for source lang)
###
if [ "$TRAIN_LM" = true ]; then
    LANG=1
    MODEL_NAME="lm$LANG"
    MAX_TIME_SECONDS=7200
    MAX_EPOCHS=1

    START_TIME_LM_1=$SECONDS
    # Running the model
    PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
                --data_path="$DATA_PATH" \
                --hp_file_path="$HP_FILE_PATH" \
                --lang="$LANG" \
                --max_epochs=$MAX_EPOCHS \
                --max_time_seconds=$MAX_TIME_SECONDS
    ELAPSED_TIME_LM_1=$(($SECONDS - $START_TIME_LM_1))

    ###
    # Running second LM model (for target lang)
    ###

    LANG=2
    MODEL_NAME="lm$LANG"
    MAX_TIME_SECONDS=7200
    MAX_EPOCHS=1
    START_TIME_LM_2=$SECONDS
    # Running the model
    PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_lm.py" "$MODEL_NAME" \
                --data_path="$DATA_PATH" \
                --hp_file_path="$HP_FILE_PATH" \
                --lang="$LANG" \
                --max_epochs=$MAX_EPOCHS \
                --max_time_seconds=$MAX_TIME_SECONDS
    ELAPSED_TIME_LM_2=$(($SECONDS - $START_TIME_LM_1))
fi

###########
# Running transformer with fused LM
###########

MODEL_NAME="transformer"
BATCH_SIZE_FOR_INFERENCE=32
MAX_TIME_SECONDS=21600
SHOULD_VALIDATE_EVERY_EPOCH=True
MAX_EPOCHS=200
USE_EARLY_STOPPING=True
EARLY_STOPPING_LAST_N=5
WARM_UP_NUM_EPOCHS=5


START_TIME_TRANS_TR=$SECONDS
# Training the model
if [ "$TRAIN_TR" = true ]; then
    PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/train_fused.py" "$MODEL_NAME" \
                --data_path="$DATA_PATH" \
                --hp_file_path="$HP_FILE_PATH" \
                --max_time_seconds="$MAX_TIME_SECONDS" \
                --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
                --use_early_stopping="$USE_EARLY_STOPPING" \
                --early_stopping_last_n="$EARLY_STOPPING_LAST_N" \
                --max_epochs="$MAX_EPOCHS" \
                --validate_every_epoch="$SHOULD_VALIDATE_EVERY_EPOCH" \
                --target_lm_path="$PROJECT_DIR/trained_models/lm2/model.npz" \
                --src_lm_path="$PROJECT_DIR/trained_models/lm1/model.npz" \
                --warm_up_num_epochs="$WARM_UP_NUM_EPOCHS"
fi

ELAPSED_TIME_TRANS_TR=$(($SECONDS - $START_TIME_TRANS_TR))
# Running the model

START_TIME_TRANS_INF=$SECONDS
PYTHONPATH="$PROJECT_DIR" python3.6 "$PROJECT_DIR/src/run_fused.py" "$MODEL_NAME" \
            --data_path="$DATA_PATH" \
            --model_path="$PROJECT_DIR/trained_models/$MODEL_NAME/model.npz" \
            --input_path="$DATA_PATH/bpe_input.txt" \
            --output_path="$DATA_PATH/output.txt" \
            --hp_file_path="$HP_FILE_PATH" \
            --batch_size_for_inference="$BATCH_SIZE_FOR_INFERENCE" \
            --target_lm_path="$PROJECT_DIR/trained_models/lm2/model.npz"

ELAPSED_TIME_TRANS_INF=$(($SECONDS - $START_TIME_TRANS_INF))

echo "$ELAPSED_TIME_LM_1"
echo "$ELAPSED_TIME_LM_2"
echo "$ELAPSED_TIME_TRANS_TR"
echo "$ELAPSED_TIME_TRANS_INF"

(echo "$ELAPSED_TIME_LM_1"; echo "$ELAPSED_TIME_LM_2";
echo "$ELAPSED_TIME_TRANS_TR"; echo "$ELAPSED_TIME_TRANS_INF") > "$PROJECT_DIR/trained_models/elapsed_time.log"
##cat $DATA_PATH/output.tok.txt | $mosesdecoder/scripts/tokenizer/detokenizer.perl > $OUTPUT_DATA_PATH/output.txt

#python3.6 final_fix.py $DATA_PATH/output.tok.txt $OUTPUT_DATA_PATH/output.txt
