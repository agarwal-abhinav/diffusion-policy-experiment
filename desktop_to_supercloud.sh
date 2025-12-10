#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="data/pretrained_models/robomimic_resnet18_240by240_spatialsoftmax_gn.pth"
# SOURCE_DIR="data/diffusion_experiments/grasp_two_bins/two_bins_flat_data_2_left_same_return_same_center_1_second_total.zarr"
# SOURCE_DIR="data/pretrained_models/robomimic_resnet18_spatialsoftmax_gn.pth"
# SOURCE_DIR="data/outputs/context_length_exp_adam_data_variable_training/random_sprinkle_2_to_obs_resnet_init/16_obs"
# SOURCE_DIR="data/pusht"
SOURCE_DIR="data/diffusion_experiments/grasp_two_bins/two_bins_flat_data_3_right_diff_return_diff_center_1_second_total.zarr"
# Target directory on the target computer
TARGET_USER="aagarwal2"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/diffusion_experiments/grasp_two_bins/"

# TARGET_USER="schia"
# TARGET_HOST="txe1-login.mit.edu"
# TARGET_DIR="/home/gridsan/schia/abhinav_workspace/gcs-diffusion/data/planar_pushing_cotrain/"

# Rsync command with -avh flags
rsync -avh "$SOURCE_DIR" "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" 

# Example usage:
# ./rsync_script.sh
