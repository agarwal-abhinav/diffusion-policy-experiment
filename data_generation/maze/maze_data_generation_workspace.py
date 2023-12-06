import numpy as np
import data_generation.maze.gcs_utils as gcs_utils

from multiprocessing import Pool
from tqdm import tqdm
from typing import List
from data_generation.maze.maze_environment import MazeEnvironment
from pydrake.geometry.optimization import (HPolyhedron,
                                           VPolytope)

class MazeDataGenerationWorkspace:
    def __init__(self):
        pass

    def run(self):
        pass