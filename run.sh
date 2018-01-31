#!/bin/bash

INPUT_DATA_PATH="/input"
OUTPUT_DATA_PATH="/output"

# Preparing data
./tokenize.sh /nmt "$INPUT_DATA_PATH"

# Training the model
PYTHONPATH=. python3.6 src/train.py gnmt \
            --data_path="$INPUT_DATA_PATH" \
            --batch_size=4 \
            --hp_file=hp_files/gnmt.json \
            --gpu_memory_fraction=0.5 \
            --validate_every=2 \
            --val_split_size=0.1 \
            --save_every=1 \
            --max_epochs=5 \
            --use_early_stopping=True

# Running the model
PYTHONPATH=. python3.6 src/run.py gnmt \
            --data_path="$INPUT_DATA_PATH" \
            --model_path=/nmt/trained_models/super_gnmt_model/model.npz \
            --input_path="$INPUT_DATA_PATH/bpe_input.txt" \
            --output_path="$OUTPUT_DATA_PATH/output.txt" \
            --gpu_memory_fraction=0.5 \
            --hp_file=hp_files/gnmt.json \
            --run_batch_size=2

