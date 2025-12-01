#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Variable_length_constant_obs_steps_resnet_init_12_obs #Set the job name to Example_SNMC_GPU
#SBATCH --time=40:00:00 #Set the wall clock limit to 15hr 30min
#SBATCH --nodes=1 #Request 1 nodes
#SBATCH --ntasks-per-node=8 #Request 16 tasks/cores per node
#SBATCH --mem=50G #Request 50 (50GB) per node
#SBATCH --output=submit_training_aces.sh.log-%j #Redirect stdout/err to file
#SBATCH --partition=gpu #Specify partition to submit job to
#SBATCH --gres=gpu:h100:1 #Specify GPU(s) per node, 1 H100 GPUs

##OPTIONAL JOB SPECIFICATIONS 
##SBATCH --mail-type=ALL #Send email on all job events
##SBATCH --mail-user=abhi_ag@mit.edu #Send all emails to email_address

module load GCCcore/11.3.0 Python/3.10.4

source /scratch/user/u.aa336018/RLG/python_environments/aces_diff_training_h100/bin/activate 

wandb offline 

##CONFIG_DIR=config/planar_pushing/context_length_exp_adam_data_variable_training/constant_obs_steps_resnet_init
##CONFIG_NAME=12_obs.yaml
##HYDRA_RUN_DIR=data/outputs/context_length_exp_adam_data_variable_training/constant_obs_steps_resnet_init/12_obs

##CONFIG_DIR=config/planar_pushing/context_length_exp_adam_data_resnet_plus_transformer/all_tokens_limited_attention/upto_12
##CONFIG_NAME=16_obs.yaml
##HYDRA_RUN_DIR=data/outputs/context_length_exp_adam_data_resnet_plus_transformer/all_tokens_limited_attention/upto_12/16_obs

CONFIG_DIR=config/long_context_planar_pushing/two_modes/unet_cross_attention/constant_obs_steps/
CONFIG_NAME=60_obs.yaml
HYDRA_RUN_DIR=data/outputs/long_context_planar_pushing/two_modes/unet_cross_attention/constant_obs_steps/60_obs

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME hydra.run.dir=$HYDRA_RUN_DIR
