#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="data/sim_sim_tee_data_carbon_large.zarr"
SOURCE_DIR="data/diffusion_experiments/grasp_two_bins/two_bins_data_1_desired_commands_only_right.zarr"

# Target directory on the target computer
TARGET_USER="aagarwal2"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/diffusion_experiments/grasp_two_bins/"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./rsync_script.sh
