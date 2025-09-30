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
# echo "[submit_training.sh] Setting wandb to offline"
# wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
export HYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running training code..."
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

# CONFIG_DIR=config/robomimic/tool_hang/2_encoder_basic_freeze/
# CONFIG_NAME=2_obs.yaml
# HYDRA_RUN_DIR=data/outputs/robomimic/tool_hang/2_encoder_basic_freeze/2_obs

POLICY_PATH=data/outputs/robomimic/square/variable_training/random_2_to_obs/16_obs/

python run_robomimic_eval.py --path $POLICY_PATH

# python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
# 	hydra.run.dir=$HYDRA_RUN_DIR
