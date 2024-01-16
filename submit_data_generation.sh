#!/bin/bash

# Usage
# LLsub ./submit_data_generation.sh [NODES, NPPN, NT]

# Initialize and Load Modules
source /etc/profile
module load anaconda/2023b
 
python data_generation/maze/generate_maze_data.py --config-name supercloud_maze_data_generation.yaml task_id=$LLSUB_RANK