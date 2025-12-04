#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/2023b

# Assume current directory is gcs-diffusion
source .robodiff/bin/activate || echo "Training with anaconda/2023b module instead of venv"

# Set wandb to offline since Supercloud has no internet access
echo "[submit_training.sh] Setting wandb to offline"
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
export cHYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running training code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

# CONFIG_DIR=config/planar_pushing/context_length_exp_adam_data_variable_training/random_sprinkle_2_to_obs_resnet_init/
# CONFIG_NAME=12_obs.yaml
# HYDRA_RUN_DIR=data/outputs/context_length_exp_adam_data_variable_training/random_sprinkle_2_to_obs_resnet_init/12_obs

# CONFIG_DIR=config/planar_pushing/diffusion_transformer_training/context_length_exp_adam_data/robomimic_resnet18_init/
# CONFIG_NAME=1_obs.yaml
# HYDRA_RUN_DIR=data/outputs/diffusion_transformer_training/context_length_exp_adam_data/robomimic_resnet18_init/1_obs

# CONFIG_DIR=config/planar_pushing/context_length_exp_adam_data_constant_model_size/
# CONFIG_NAME=5_obs.yaml
# HYDRA_RUN_DIR=data/outputs/context_length_exp_adam_data_constant_model_size/5_obs

# CONFIG_DIR=config/grasp_two_bins_flat/attention_training/random_4_to_obs
# CONFIG_NAME=22_obs.yaml
# HYDRA_RUN_DIR=data/outputs/grasp_two_bins_flat/attention_training/random_4_to_obs/22_obs

CONFIG_DIR=config/grasp_two_bins_flat_green/basic_training/
CONFIG_NAME=28_obs.yaml
HYDRA_RUN_DIR=data/outputs/grasp_two_bins_flat_green/basic_training/28_obs

# CONFIG_DIR=config/canonical_planar_pushing/initial_training/
# CONFIG_NAME=20_obs_h_32.yaml
# HYDRA_RUN_DIR=data/outputs/canonical_planar_pushing/initial_training/20_obs_h_32

# CONFIG_DIR=config/long_context_planar_pushing/two_modes/unet_film/
# CONFIG_NAME=10_obs.yaml
# HYDRA_RUN_DIR=data/outputs/long_context_planar_pushing/two_modes/unet_film/10_obs

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
