#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="data/sim_sim_tee_data_carbon_large.zarr"
SOURCE_DIR="data/canonical_planar_pushing_lc/lc_block_push_combined.zarr"

# Target directory on the target computer
TARGET_USER="aagarwal2"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/canonical_planar_pushing_lc/"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./rsync_script.sh
