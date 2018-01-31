#!/bin/bash

DATA_PATH="/nmt/data"
INPUT_DATA_PATH="/data"
OUTPUT_DATA_PATH="/output"

# What the hack is this?
cd /nmt

# Preparing data
/nmt/tokenize.sh /nmt "$INPUT_DATA_PATH"

# Training the model
PYTHONPATH=/nmt python3.6 /nmt/src/train.py transformer \
            --data_path="$DATA_PATH" \
            --hp_file_path=/nmt/hp_files/mini_transformer.json \
            --validate_every=100 \
            --max_time_seconds=21600 \
            --use_early_stopping=True \
            --early_stopping_last_n=5

# Running the model
PYTHONPATH=/nmt python3.6 /nmt/src/run.py transformer \
            --data_path="$DATA_PATH" \
            --model_path=/nmt/trained_models/transformer/model.npz \
            --input_path="$DATA_PATH/bpe_input.txt" \
            --output_path="$OUTPUT_DATA_PATH/output.txt" \
            --hp_file_path=/nmt/hp_files/mini_transformer.json \
            --batch_size_on_validation=64
