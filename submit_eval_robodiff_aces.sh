#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Variable_length_constant_obs_steps_resnet_init_12_obs #Set the job name to Example_SNMC_GPU
#SBATCH --time=23:00:00 #Set the wall clock limit to 15hr 30min
#SBATCH --nodes=1 #Request 1 nodes
#SBATCH --ntasks-per-node=8 #Request 16 tasks/cores per node
#SBATCH --mem=45G #Request 40 (40GB) per node
#SBATCH --output=submit_training_aces.sh.log-%j #Redirect stdout/err to file
#SBATCH --partition=gpu #Specify partition to submit job to
#SBATCH --gres=gpu:h100:1 #Specify GPU(s) per node, 1 H100 GPUs

##OPTIONAL JOB SPECIFICATIONS 
##SBATCH --mail-type=ALL #Send email on all job events
##SBATCH --mail-user=abhi_ag@mit.edu #Send all emails to email_address

module load Anaconda3/2023.07-2 CUDA/12.4.0

source activate robodiff-h100

wandb offline 


POLICY_PATH=data/outputs/robomimic/tool_hang/variable_training/random_2_to_obs/2_obs/

python run_robomimic_eval.py --path $POLICY_PATH