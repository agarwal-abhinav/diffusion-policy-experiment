#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="outputs/diffusion_experiments/cartpole_controller_noise_100/partially_observable_h_50_pred_8_ablation/3_traj/"
# SOURCE_DIR="outputs/diffusion_experiments/random_system_seed_0_controller_noise_0.5/partially_observable_ss/0_mean_50_context/checkpoints/"
SOURCE_DIR="data/outputs/context_length_exp_adam_data_constant_model_size_init_encoder"
# SOURCE_DIR="data/outputs/"

# Target directory on the target computer
TARGET_USER="aagarwal2"
TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/context_length_exp_adam_data_constant_model_size_init_encoder/16_encoder_freeze"
# TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/context_length_exp_adam_data_constant_model_size_init_encoder/2_encoder_init/12_obs/normalizer.pt"
# TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/context_length_exp_adam_data_constant_model_size_low_pass"

# TARGET_DIR="/home/gridsan/aagarwal2/RLG/diffusion-search-learning/outputs/diffusion_experiments/random_system_seed_0_controller_noise_0.5/partially_observable_ss/0_mean_50_context/checkpoints/latest.ckpt"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR" 

# Example usage:
# ./rsync_script.sh


