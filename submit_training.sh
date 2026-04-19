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
export HYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running training code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_96/mode_4_0
# CONFIG_NAME=8_obs.yaml 
# HYDRA_RUN_DIR=data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_96/mode_4_0/8_obs

# CONFIG_DIR=config/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_48
# CONFIG_NAME=8_obs.yaml 
# HYDRA_RUN_DIR=data/outputs/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_48/8_obs

CONFIG_DIR=config/iros/planar_pushing/variable_context/data_80
CONFIG_NAME=variable_progressive_resnet_full.yaml 
HYDRA_RUN_DIR=data/outputs/iros/planar_pushing/variable_context/data_80/variable_progressive_resnet_full

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR logging.mode="offline"
