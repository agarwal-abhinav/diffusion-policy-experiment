import numpy as np
import time
import pickle
import zarr
from datetime import datetime
from multiprocessing import Pool
import data_generation.maze.gcs_utils as gcs_utils

from multiprocessing import Pool
from tqdm import tqdm
from typing import List
from data_generation.maze.maze_environment import MazeEnvironment, MazeEnvironmentGenerator
from pydrake.geometry.optimization import (HPolyhedron,
                                           VPolytope,
                                           Point,
                                           GraphOfConvexSetsOptions)
from pydrake.solvers import MosekSolver
from pydrake.planning import GcsTrajectoryOptimization

class MazeDataGenerationWorkspace:
    def __init__(self, maze_generator: MazeEnvironmentGenerator,
                  num_mazes: int, 
                  num_trajectories_per_maze: int,
                  num_processes: int,
                  # GCS settings
                  max_rounded_paths: int=3, # number of GCS rounded paths
                  max_velocity: float=1.0,
                  continuity_order: int=1,
                  bezier_order: int =3):
        if num_trajectories_per_maze % num_processes != 0:
            raise ValueError("num_trajectories_per_maze must be divisible by num_processes")

        self.maze_generator = maze_generator
        self.num_mazes = num_mazes
        self.num_trajectories_per_maze = num_trajectories_per_maze
        self.num_processes = num_processes
        # GCS settings
        self.max_rounded_paths = max_rounded_paths
        self.max_velocity = max_velocity
        self.continuity_order = continuity_order
        self.bezier_order = bezier_order

    def run(self):
        mazes = [self.maze_generator.generate_maze() \
                 for i in range(self.num_mazes)]
        data = {'mazes': mazes, 
                'sources': [], 
                'targets': [], 
                'trajectories': []}
        for maze in tqdm(mazes, position=0):
            data['sources'].append([])
            data['targets'].append([])
            data['trajectories'].append([])

            with Pool(self.num_processes) as p:
                pooled_data = \
                    p.map(self.generate_data, [maze]*self.num_processes)
            for d in pooled_data:
                data['sources'][-1].extend(d['sources'])
                data['targets'][-1].extend(d['targets'])
                data['trajectories'][-1].extend(d['trajectories'])
        
        # Save with pickle
        # TODO: save with Zarr
        # TODO: save to specific directory
        with open('maze_data.pkl', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def generate_data(self, maze: MazeEnvironment):
        # TODO: save the images!
        dim = maze.dim
        bounds = maze.bounds

        # Build graph
        start_time = time.time()
        gcs = GcsTrajectoryOptimization(dim)
        free_space = gcs.AddRegions(maze.regions)

        # Cost & Constraints
        gcs.AddVelocityBounds(np.array([-self.max_velocity, self.max_velocity]),
                            np.array([-self.max_velocity, self.max_velocity]))
        gcs.AddPathContinuityConstraints(self.continuity_order)
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        time_to_build_graph = time.time() - start_time

        # Configure Options
        options = GraphOfConvexSetsOptions()
        options.max_rounded_paths = self.max_rounded_paths

        # Generate seed for thread
        seed = int((datetime.now().timestamp() % 1) * 1e6)
        np.random.seed(seed)

        # Generate data
        data = {'sources': [], 'targets': [], 'trajectories': []}
        pbar = tqdm(total=self.num_trajectories_per_maze // 5)
        while len(data['trajectories'] < self.num_trajectories_per_maze):
            start_time = time.time()
            start = maze.sample_start_point()
            goal = maze.sample_end_point()
            source = gcs.AddRegions([Point(start)], 0)
            target = gcs.AddRegions([Point(goal)], 0)
            gcs.AddEdges(source, free_space)
            gcs.AddEdges(free_space, target)

            [traj, result] = gcs.SolvePath(source, target, options)
            if result.is_success():
                data['sources'].append(start)
                data['targets'].append(goal)
                data['trajectories'].append(
                    gcs_utils.composite_trajectory_to_array(traj).transpose()
                )
                pbar.update(1)

            # Rebuild GCS object if solve times become too slow
            end_time = time.time()
            factor = 1.5
            if end_time-start_time > factor * time_to_build_graph:
                gcs = GcsTrajectoryOptimization(dim)
                free_space = gcs.AddRegions(maze.regions)
                gcs.AddVelocityBounds(np.array([-self.max_velocity, self.max_velocity]),
                                    np.array([-self.max_velocity, self.max_velocity]))
                gcs.AddPathContinuityConstraints(self.continuity_order)
                gcs.AddTimeCost()
                gcs.AddPathLengthCost()
        
        pbar.close()
        return data
    
if __name__ == "__name__":
    pass