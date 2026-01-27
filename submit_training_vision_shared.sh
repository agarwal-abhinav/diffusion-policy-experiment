#!/bin/bash

#SBATCH --job-name=1_mode_8_obs_d_48
#SBATCH --time=23:59:00 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=64G 
#SBATCH --output=submit_training_vision_shared.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=vision-shared-rtx4090
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --requeue

# vision-shared-h100, vision-shared-l40s, vision-shared-a100 are alternatives 
# vision-shared-rtx4090 works for smaller policies
# csail-shared-h200 also seems largely empty
# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment"

echo "NODE: $SLURMD_NODENAME"
echo "JOB:  $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Exporting home directory, sourcing conda, and activating conda environment 
export HOME=/data/locomotion/abhi_ag/
source /data/scratch-oc40/abhi_ag/python_environments/miniconda3/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

# activate the conda environment 
conda activate gcs-diffusion-training

echo "[submit_training.sh] PyTorch and CUDA versions:"

python - << 'EOF'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available())
EOF

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

CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_48/mode_4
CONFIG_NAME=8_obs.yaml
HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_48/mode_4/8_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_24/mode_4_0
# CONFIG_NAME=72_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_24/mode_4_0/72_obs

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
