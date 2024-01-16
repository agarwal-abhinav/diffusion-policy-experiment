"""
Usage: From root directory, run
python data_generation/maze/generate_maze_data.py --config-name maze_data_generation.yaml
"""

import hydra
import numpy as np
import sys
import shutil
import pathlib
from omegaconf import OmegaConf

from data_generation.maze.maze_data_generation_workspace import MazeDataGenerationWorkspace
from data_generation.maze.maze_environment_generator import MazeEnvironmentGenerator

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'config'))
)
def main(cfg: OmegaConf):
    # Generate data
    gen_cfg = cfg.environment_generator
    data_dir = cfg.data_dir
    if cfg.task_id is not None:
        data_dir = f'{data_dir}/{cfg.task_id}'
    maze_generator = MazeEnvironmentGenerator(
        min_num_obstacles=gen_cfg.min_num_obstacles,
        max_num_obstacles=gen_cfg.max_num_obstacles,
        min_obstacle_width=gen_cfg.min_obstacle_width,
        max_obstacle_width=gen_cfg.max_obstacle_width,
        min_obstacle_height=gen_cfg.min_obstacle_height,
        max_obstacle_height=gen_cfg.max_obstacle_height,
        border_padding=gen_cfg.border_padding,
        bounds=np.array(gen_cfg.bounds),
        non_overlapping_centers=gen_cfg.non_overlapping_centers
    )
    maze_data_generation_workspace = MazeDataGenerationWorkspace(
        maze_generator=maze_generator,
        num_mazes_per_proc=cfg.num_mazes_per_proc,
        num_trajectories_per_maze=cfg.num_trajectories_per_maze,
        num_processes=cfg.num_processes,
        data_dir=data_dir,
        append_date_time=cfg.append_date_time,
        max_rounded_paths=cfg.max_rounded_paths,
        max_velocity=cfg.max_velocity,
        continuity_order=cfg.continuity_order,
        bezier_order=cfg.bezier_order
    )
    maze_data_generation_workspace.run()

    # Add config file to data directory
    config_name = sys.argv[2]
    src = str(pathlib.Path(__file__).parent.parent.joinpath(
        'config', config_name))
    dst = str(pathlib.Path(__file__).parent.parent.parent.joinpath(
        maze_data_generation_workspace.data_dir, 'config.yaml'))
    shutil.copy(src, dst)
    

if __name__ == "__main__":
    main()