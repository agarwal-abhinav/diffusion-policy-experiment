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

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'config'))
)
def main(cfg: OmegaConf):
    # Generate data
    cls = hydra.utils.get_class(cfg._target_)
    maze_data_generation_workspace = cls(cfg=cfg)
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