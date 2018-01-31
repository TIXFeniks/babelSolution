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
            --batch_size=64 \
            --hp_file=/nmt/hp_files/mini_transformer.json \
            --gpu_memory_fraction=1 \
            --validate_every=1000 \
            --save_every=500 \
            --max_epochs=100 \
            --use_early_stopping=True \
            --early_stopping_last_n=10

# Running the model
PYTHONPATH=/nmt python3.6 /nmt/src/run.py transformer \
            --data_path="$DATA_PATH" \
            --model_path=/nmt/trained_models/transformer/model.npz \
            --input_path="$DATA_PATH/bpe_input.txt" \
            --output_path="$OUTPUT_DATA_PATH/output.txt" \
            --gpu_memory_fraction=1 \
            --hp_file=/nmt/hp_files/mini_transformer.json \
            --run_batch_size=64
