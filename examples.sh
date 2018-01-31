PYTHONPATH=. python src/train.py gnmt \
            --data_path=data \
            --batch_size=4 \
            --hp_file=hp_files/gnmt.json \
            --gpu_memory_fraction=0.3 \
            --validate_every=100 \
            --val_split_size=0.1 \
            --save_every=100

# Configuring docker
sudo nvidia-docker run -v /home/user32878/data/test_data2/:/input -v /home/user32878/data/test_data2/:/output -it universome/nmt /nmt/run.sh
sudo nvidia-docker build -t universome/nmt .
sudo docker stop $(sudo docker ps -a -q) && sudo docker rm $(sudo docker ps -a -q)

# Copying files to remote to run experiments
scp -r MY_FILE dh:~/nmt
