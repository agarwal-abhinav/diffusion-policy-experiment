#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="data/sim_sim_tee_data_carbon_large.zarr"
SOURCE_DIR="data/outputs/context_length_exp_adam_data_constant_model_size/16_obs/best_ckpt_encoder.pth"

# Target directory on the target computer
TARGET_USER="aagarwal2"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/context_length_exp_adam_data_constant_model_size/16_obs/"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./rsync_script.sh
