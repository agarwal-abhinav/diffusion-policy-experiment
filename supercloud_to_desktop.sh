#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="outputs/diffusion_experiments/cartpole_controller_noise_100/partially_observable_h_50_pred_8_ablation/3_traj/"
# SOURCE_DIR="outputs/diffusion_experiments/random_system_seed_0_controller_noise_0.5/partially_observable_ss/0_mean_50_context/checkpoints/"
# SOURCE_DIR="data/outputs/grasp_two_bins/constant_model_size_frozen_encoder/2_frozen/5_obs/checkpoints/"
# SOURCE_DIR="data/outputs/planar_pushing/diffusion_transformer_training/context_length_exp_adam_data/"
# SOURCE_DIR="data/outputs/context_length_exp_adam_data_resnet_plus_transformer/cls_token_only_2_encoder_freeze_regular_policy/"
SOURCE_DIR="data/outputs/context_length_exp_adam_data_resnet_plus_transformer/cls_token_only_causal/"

# Target directory on the target computer
TARGET_USER="aagarwal2"
TARGET_HOST="txe1-login.mit.edu"
# TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/grasp_two_bins/constant_model_size_frozen_encoder/2_frozen/5_obs/checkpoints/latest.ckpt"
# TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/planar_pushing/diffusion_transformer_training/context_length_exp_adam_data/5_obs"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/context_length_exp_adam_data_resnet_plus_transformer/cls_token_only_causal/12_obs"

# TARGET_DIR="/home/gridsan/aagarwal2/RLG/diffusion-search-learning/outputs/diffusion_experiments/random_system_seed_0_controller_noise_0.5/partially_observable_ss/0_mean_50_context/checkpoints/latest.ckpt"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR" 

# Example usage:
# ./rsync_script.sh


