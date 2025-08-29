#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="data/sim_sim_tee_data_carbon_large.zarr"
# SOURCE_DIR="data/diffusion_experiments/grasp_two_bins/two_bins_flat_data_2_right.zarr"
SOURCE_DIR="data/pretrained_models/mc3_18.pth"

# Target directory on the target computer
TARGET_USER="aagarwal2"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/pretrained_models/"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./rsync_script.sh
