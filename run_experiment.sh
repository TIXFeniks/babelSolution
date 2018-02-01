#!/bin/bash

EXPERIMENT_NAME=$1

# Fetching the branch
git clone https://github.com/TIXFeniks/babelSolution "$EXPERIMENT_NAME"
cd "$EXPERIMENT_NAME"
git checkout "$EXPERIMENT_NAME"

# Running experiment
# 1. Build docker image
sudo nvidia-docker build -t "universome/$EXPERIMENT_NAME" .

# 2. OPTIONAL: Testing locally on small data
# 2.1. Creating output dir
OUTPUT_DIR="/home/user32878/data/outputs/output_$EXPERIMENT_NAME"
mkdir "$OUTPUT_DIR"

# 2.2 Running experiment locally on small data
sudo nvidia-docker run -v /home/user32878/data/test_data2/:/data -v "$OUTPUT_DIR":/output -it "universome/$EXPERIMENT_NAME" /nmt/run_lm_fused.sh

# 3. Pushing to docker hub
sudo docker push "universome/$EXPERIMENT_NAME"

# Let's cat metadata.json which can be copypasted
METADATA="{\"image\": \"universome/$EXPERIMENT_NAME\", \"entry_point\": \"/nmt/run_lm_fused.sh\"}"
echo $METADATA

cd ..
rm -rf "$EXPERIMENT_NAME"

# In case of there are too many docker containers
# sudo docker stop $(sudo docker ps -a -q) && sudo docker rm $(sudo docker ps -a -q)

# Metadata json should look like this
# {
#     "image": "universome/"$EXPERIMENT_NAME"",
#     "entry_point": "/nmt/run_lm_fused.sh"
# }
