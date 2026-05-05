#!/bin/bash

#SBATCH --job-name=eval_robomimic
#SBATCH --time=23:59:00 
#SBATCH --cpus-per-task=26
#SBATCH --mem=70G 
#SBATCH --output=submit_eval_robodiff_csail_shared.sh.log-%j
#SBATCH --account=locomotion 
#SBATCH --partition=vision-shared-rtx3080
#SBATCH --qos=shared-if-available
#SBATCH --gres=gpu:1
#SBATCH --requeue

echo "[submit_training.sh] Loading modules and virtual environment"

echo "NODE: $SLURMD_NODENAME"
echo "JOB:  $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

# Exporting home directory, sourcing conda, and activating conda environment 
export HOME=/data/locomotion/abhi_ag/
source /data/locomotion/abhi_ag/miniconda3/etc/profile.d/conda.sh

# activate the conda environment 
conda activate robodiff
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

POLICY_PATH=data/outputs/iros/robomimic/square/unet_cross_attention/data_100/20_obs_npast8
OUTPUT_ROOT=data/eval_output/iros/robomimic/square/unet_cross_attention/data_100/20_obs_npast8

# POLICY_PATH=data/outputs/iros/robomimic/square/variable_context/data_50/variable_random_sprinkle_resnet_full
# OUTPUT_ROOT=data/eval_output/iros/robomimic/square/variable_context/data_50/variable_random_sprinkle_resnet_full

python run_robomimic_eval.py --path $POLICY_PATH --eval-script eval.py --output-root $OUTPUT_ROOT
