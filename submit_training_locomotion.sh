#!/bin/bash

#SBATCH --job-name=2_mode_32_obs_d_48_dit_light
#SBATCH --time=55:00:00 
#SBATCH --cpus-per-task=20 
#SBATCH --mem=90G 
#SBATCH --output=submit_training_locomotion.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=locomotion-h200 
#SBATCH --qos=locomotion-main
#SBATCH --gres=gpu:1

# name for long context pushing: 2_mode_80_obs_d_24
# name for long context grasping: gr_d_12_obs_24
# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment"

echo "NODE: $SLURMD_NODENAME"
echo "JOB:  $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Exporting home directory, sourcing conda, and activating conda environment 
# currently home is set to where it should be and python is installed in scratch 
# porting this python to new home is the next goal 
export HOME=/data/locomotion/abhi_ag/
source /data/scratch-oc40/abhi_ag/python_environments/miniconda3/etc/profile.d/conda.sh

# activate the conda environment 
conda activate gcs-diffusion-training
export PYTHONNOUSERSITE=1

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
# CONFIG_NAME=80_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_24/mode_4/80_obs

CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_48/mode_4_0_light_model
CONFIG_NAME=32_obs.yaml
HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_48/mode_4_0_light_model/32_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0
# CONFIG_NAME=80_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0/80_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_24/mode_4_0
# CONFIG_NAME=80_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0
# CONFIG_NAME=31_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0/31_obs

# CONFIG_DIR=config/iros/long_context_grasping/data_experiments/unet_film/two_modes_diff_center_diff_return/data_24/
# CONFIG_NAME=8_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_grasping/data_experiments/unet_film/two_modes_diff_center_diff_return/data_24/8_obs

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
