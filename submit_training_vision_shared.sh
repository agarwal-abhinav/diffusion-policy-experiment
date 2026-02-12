#!/bin/bash

#SBATCH --job-name=2_mode_40_obs_d_48_constant_then_skip_second_frame
#SBATCH --time=23:59:00 
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G 
#SBATCH --output=submit_training_vision_shared.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=csail-shared-h200
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --requeue

# job names for long context pushing: 1_mode_72_obs_d_72, 2_mode_40_obs_d_48_constant_then_skip_second_frame
# job names for long context grasping: gr_d_24_obs_20
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

# activate the conda environment 
conda activate gcs-diffusion-training
export PYTHONNOUSERSITE=1

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

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_24/mode_4
# CONFIG_NAME=60_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_24/mode_4/60_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_96/mode_4_0
# CONFIG_NAME=32_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_96/mode_4_0/32_obs

# CONFIG_DIR=config/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_24/
# CONFIG_NAME=20_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_24/20_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0
# CONFIG_NAME=3_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0/3_obs

CONFIG_DIR=config/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/constant_then_skip_second_frame_mode_4_0
CONFIG_NAME=40_obs.yaml
HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/constant_then_skip_second_frame_mode_4_0/40_obs

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
