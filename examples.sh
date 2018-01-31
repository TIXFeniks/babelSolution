PYTHONPATH=. python src/train.py gnmt \
            --data_path=data \
            --batch_size=4 \
            --hp_file=hp_files/gnmt.json \
            --gpu_memory_fraction=0.3 \
            --validate_every=100 \
            --val_split_size=0.1 \
            --save_every=100
