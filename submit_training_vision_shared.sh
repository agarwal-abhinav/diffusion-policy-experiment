#!/bin/bash

#SBATCH --job-name=rl_train
#SBATCH --time=23:59:00 
#SBATCH --cpus-per-task=24
#SBATCH --mem=50G 
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
source /data/locomotion/abhi_ag/miniconda3/etc/profile.d/conda.sh

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

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_96/mode_4_0
# CONFIG_NAME=80_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_film/two_modes/data_96/mode_4_0/80_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_96/mode_4_0
# CONFIG_NAME=72_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_96/mode_4_0/72_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/vit_cross_attention/two_modes/data_48/mode_4_0
# CONFIG_NAME=80_obs_finetune_vit_a_small_shared_npast16.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/vit_cross_attention/two_modes/data_48/mode_4_0/80_obs_finetune_vit_a_small_shared_npast16

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/vit_cross_attention/two_modes/data_48/mode_4_0
# CONFIG_NAME=80_obs_frozen_vit_b_per_stream_spatial_k64_full.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/data_experiments/vit_cross_attention/two_modes/data_48/mode_4_0/80_obs_frozen_vit_b_per_stream_spatial_k64_full

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0
# CONFIG_NAME=48_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/four_modes/data_192/mode_4_0/48_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_24/mode_4_0
# CONFIG_NAME=4_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/resnet18_init/unet_cross_attention/two_modes/data_24/mode_4_0/4_obs

# CONFIG_DIR=config/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_24/
# CONFIG_NAME=20_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_24/20_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0
# CONFIG_NAME=3_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/recent_plus_72_mode_4_0/3_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/constant_then_skip_second_frame_mode_4_0
# CONFIG_NAME=40_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/skip_frame_study/unet_cross_attention/two_modes/data_48/constant_then_skip_second_frame_mode_4_0/40_obs

# CONFIG_DIR=config/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_24/
# CONFIG_NAME=16_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_grasping/data_experiments/unet_cross_attention/two_modes_diff_center_diff_return/data_24/16_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/variable_context/two_modes/data_48/mode_4_0
# CONFIG_NAME=variable_random_sprinkle_resnet_npast16.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/long_context_planar_pushing/variable_context/two_modes/data_48/mode_4_0/variable_random_sprinkle_resnet_npast16

CONFIG_DIR=config/iros/robomimic/lift/unet_cross_attention/data_100
CONFIG_NAME=16_obs.yaml
HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/robomimic/lift/unet_cross_attention/data_100/16_obs

# CONFIG_DIR=config/iros/robomimic/square/limited_past/unet_cross_attention/data_50
# CONFIG_NAME=5_obs_no_past.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/diffusion-policy-experiment/data/outputs/iros/robomimic/square/limited_past/unet_cross_attention/data_50/5_obs_no_past


python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR


# python run_robomimic_eval.py --path data/outputs/iros/robomimic/square/unet_cross_attention/data_100/5_obs/ --eval-script eval.py --output-root outputs/test