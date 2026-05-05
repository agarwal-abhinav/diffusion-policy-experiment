#!/bin/bash

#SBATCH --job-name=rs_np_train
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
source /data/locomotion/abhi_ag/miniconda3/etc/profile.d/conda.sh

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

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_48/mode_4_0_light_model
# CONFIG_NAME=32_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/dit_cross_attention/two_modes/data_48/mode_4_0_light_model/32_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/vit_cross_attention/two_modes/data_48/mode_4_0
# CONFIG_NAME=80_obs_finetune_vit_b_per_stream_proj64_shared_npast16.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/vit_cross_attention/two_modes/data_48/mode_4_0/80_obs_finetune_vit_b_per_stream_proj64_shared_npast16

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0
# CONFIG_NAME=72_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0/72_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0
# CONFIG_NAME=80_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0/80_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_24/mode_4_0
# CONFIG_NAME=80_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_24/mode_4_0/80_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0
# CONFIG_NAME=31_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0/31_obs

# CONFIG_DIR=config/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_12/
# CONFIG_NAME=48_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_12/48_obs

# CONFIG_DIR=config/iros/long_context_grasping/variable_context_ablation/two_modes_diff_center_diff_return/data_12
# CONFIG_NAME=prog_min8_max48_cur50k.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_grasping/variable_context_ablation/two_modes_diff_center_diff_return/data_12/prog_min8_max48_cur50k

# CONFIG_DIR=config/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_48/mode_4_0
# CONFIG_NAME=8_obs_no_past.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/limited_past/unet_cross_attention/two_modes/data_48/mode_4_0/8_obs_no_past

# CONFIG_DIR=config/iros/long_context_planar_pushing/variable_context/two_modes/data_24/mode_4_0
# CONFIG_NAME=variable_progressive_resnet_npast16.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/variable_context/two_modes/data_24/mode_4_0/variable_progressive_resnet_npast16

CONFIG_DIR=config/iros/robomimic/square/limited_past/unet_cross_attention/data_50
CONFIG_NAME=20_obs_no_past.yaml
HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/robomimic/square/limited_past/unet_cross_attention/data_50/20_obs_no_past

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
