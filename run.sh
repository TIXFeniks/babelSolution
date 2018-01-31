#!/bin/bash

DATA_PATH="/nmt/data"
INPUT_DATA_PATH="/input"
OUTPUT_DATA_PATH="/output"

# Preparing data
/nmt/tokenize.sh /nmt "$INPUT_DATA_PATH"

# Training the model
PYTHONPATH=/nmt python3.6 /nmt/src/train.py gnmt \
            --data_path="$DATA_PATH" \
            --batch_size=64 \
            --hp_file=/nmt/hp_files/gnmt.json \
            --gpu_memory_fraction=1 \
            --validate_every=100 \
            --save_every=100 \
            --max_epochs=10 \
            --use_early_stopping=True \
            --early_stopping_last_n=10

# Running the model
PYTHONPATH=/nmt python3.6 /nmt/src/run.py gnmt \
            --data_path="$DATA_PATH" \
            --model_path=/nmt/trained_models/super_gnmt_model/model.npz \
            --input_path="$DATA_PATH/bpe_input.txt" \
            --output_path="$OUTPUT_DATA_PATH/output.txt" \
            --gpu_memory_fraction=1 \
            --hp_file=/nmt/hp_files/gnmt.json \
            --run_batch_size=64
