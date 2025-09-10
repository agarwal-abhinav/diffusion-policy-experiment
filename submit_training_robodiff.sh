#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:1

# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/2023b
source activate robodiff

export PYTHONNOUSERSITE=True
# Assume current directory is gcs-diffusion
# source .robodiff/bin/activate || echo "Training with anaconda/2023b module instead of venv"

# Set wandb to offline since Supercloud has no internet access
echo "[submit_training.sh] Setting wandb to offline"
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
export HYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running training code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

CONFIG_DIR=config/robomimic/tool_hang/basic_training/
CONFIG_NAME=16_obs.yaml
HYDRA_RUN_DIR=data/outputs/robomimic/basic_training/16_obs

# CONFIG_DIR=config/planar_pushing/context_length_exp_adam_data_constant_model_size/robomimic_resnet18_freeze/
# CONFIG_NAME=1_obs.yaml
# HYDRA_RUN_DIR=data/outputs/context_length_exp_adam_data_constant_model_size/robomimic_resnet18_freeze/1_obs

# CONFIG_DIR=config/grasp_two_bins_flat/same_middle_same_return/basic_training
# CONFIG_NAME=22_obs.yaml
# HYDRA_RUN_DIR=data/outputs/grasp_two_bins_flat/same_middle_same_return/basic_training/22_obs

# CONFIG_DIR=config/grasp_two_bins_flat/resnet_plus_transformer/cls_token_only/
# CONFIG_NAME=24_obs.yaml
# HYDRA_RUN_DIR=data/outputs/grasp_two_bins_flat/resnet_plus_transformer/cls_token_only/24_obs

# CONFIG_DIR=config/canonical_planar_pushing/initial_training/
# CONFIG_NAME=20_obs_h_32.yaml
# HYDRA_RUN_DIR=data/outputs/canonical_planar_pushing/initial_training/20_obs_h_32

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
