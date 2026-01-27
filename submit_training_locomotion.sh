#!/bin/bash

#SBATCH --job-name=2_mode_72_obs_d_48 
#SBATCH --time=40:00:00 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=64G 
#SBATCH --output=submit_training_locomotion.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=locomotion-h200 
#SBATCH --qos=locomotion-main
#SBATCH --gres=gpu:1

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
export PYTHONNOUSERSITE=1

# activate the conda environment 
conda activate gcs-diffusion-training

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

# CONFIG_DIR=config/grasp_two_bins_flat_green/attention_training/constant_obs_steps_baseline/
# CONFIG_NAME=30_obs.yaml
# HYDRA_RUN_DIR=data/outputs/grasp_two_bins_flat_green/attention_training/constant_obs_steps_baseline/30_obs

# CONFIG_DIR=config/canonical_planar_pushing/initial_training/
# CONFIG_NAME=20_obs_h_32.yaml
# HYDRA_RUN_DIR=data/outputs/canonical_planar_pushing/initial_training/20_obs_h_32

# CONFIG_DIR=config/long_context_planar_pushing/two_modes/unet_film/0_via_mirror
# CONFIG_NAME=30_obs.yaml
# HYDRA_RUN_DIR=data/outputs/long_context_planar_pushing/two_modes/unet_film/0_via_mirror/30_obs_retry

# CONFIG_DIR=config/long_context_planar_pushing/two_modes/unet_cross_attention/constant_obs_steps_0_mirror
# CONFIG_NAME=5_obs.yaml
# HYDRA_RUN_DIR=data/outputs/long_context_planar_pushing/two_modes/unet_cross_attention/constant_obs_steps_0_mirror_25_each/5_obs

# CONFIG_DIR=config/iros/planar_pushing/data_experiments/unet_cross_attention/data_80
# CONFIG_NAME=1_obs.yaml
# HYDRA_RUN_DIR=data/outputs/iros/planar_pushing/data_experiments/unet_cross_attention/data_80/1_obs

# CONFIG_DIR=config/long_context_planar_pushing/single_mode/unet_cross_attention/4_original/constant_obs_steps
# CONFIG_NAME=5_obs.yaml
# HYDRA_RUN_DIR=data/outputs/long_context_planar_pushing/single_mode/unet_film/4_original/constant_obs_steps_25_data/5_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_24/mode_4
# CONFIG_NAME=8_obs.yaml
# HYDRA_RUN_DIR=data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_24/mode_4/8_obs

# CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_24/mode_4
# CONFIG_NAME=80_obs.yaml
# HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/single_mode/data_24/mode_4/80_obs

CONFIG_DIR=config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0
CONFIG_NAME=72_obs.yaml
HYDRA_RUN_DIR=/data/locomotion/abhi_ag/workspace/gcs-diffusion/data/outputs/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/two_modes/data_48/mode_4_0/72_obs

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR
