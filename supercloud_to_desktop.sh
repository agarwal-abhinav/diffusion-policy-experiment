#!/bin/bash

# Source directory on your computer
# SOURCE_DIR="outputs/diffusion_experiments/cartpole_controller_noise_100/partially_observable_h_50_pred_8_ablation/3_traj/"
# SOURCE_DIR="outputs/diffusion_experiments/random_system_seed_0_controller_noise_0.5/partially_observable_ss/0_mean_50_context/checkpoints/"
# SOURCE_DIR="data/outputs/grasp_two_bins_flat/same_middle_same_return/attention_training/random_4_to_obs/22_obs/"
SOURCE_DIR="data/outputs/long_context_planar_pushing/two_modes/unet_cross_attention/constant_obs_steps_0_mirror_25_each/5_obs/checkpoints/"
# SOURCE_DIR="data/outputs/long_context_planar_pushing/single_mode/unet_cross_attention/4_original/constant_obs_steps_25_data/5_obs/"

# SOURCE_DIR="data/outputs/planar_pushing/diffusion_transformer_training/context_length_exp_adam_data/"
# SOURCE_DIR="data/outputs/context_length_exp_adam_data_resnet_plus_transformer/cls_token_only_2_encoder_freeze_regular_policy/"
# SOURCE_DIR="data/outputs/robomimic/square/resnet18_init/"

# Target directory on the target computer
TARGET_USER="aagarwal2"
# TARGET_USER="schia"

TARGET_HOST="txe1-login.mit.edu"
TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/long_context_planar_pushing/two_modes/unet_cross_attention/constant_obs_steps_0_mirror_25_each/5_obs/checkpoints/latest.ckpt"
# TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/robomimic/square/resnet18_init/2_obs"
# TARGET_DIR="/home/gridsan/schia/abhinav_workspace/gcs-diffusion/data/outputs/grasp_two_bins_flat/same_middle_same_return/basic_training/24_obs/checkpoints/latest.ckpt"

# TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/planar_pushing/diffusion_transformer_training/context_length_exp_adam_data/5_obs"
# TARGET_DIR="/home/gridsan/aagarwal2/RLG/gcs-diffusion/data/outputs/context_length_exp_adam_data_resnet_plus_transformer/cls_token_only_2_encoder_freeze_regular_policy/10_obs"

# TARGET_DIR="/home/gridsan/aagarwal2/RLG/diffusion-search-learning/outputs/diffusion_experiments/random_system_seed_0_controller_noise_0.5/partially_observable_ss/0_mean_50_context/checkpoints/latest.ckpt"

# Rsync command with -avh flags
mkdir -p "$SOURCE_DIR"
rsync -avh "$TARGET_USER@$TARGET_HOST:$TARGET_DIR" "$SOURCE_DIR" 

# Example usage:
# ./rsync_script.sh


