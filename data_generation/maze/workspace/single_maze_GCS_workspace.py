import numpy as np
import time
import os
import pickle
import zarr
from datetime import datetime
from multiprocessing import Pool
import matplotlib.pyplot as plt
import logging

from multiprocessing import Pool
from tqdm import tqdm
from typing import List
from omegaconf import OmegaConf

import data_generation.maze.gcs_utils as gcs_utils
from data_generation.maze.maze_environment import MazeEnvironment
from data_generation.maze.maze_environment_generator import MazeEnvironmentGenerator
from pydrake.geometry.optimization import (HPolyhedron,
                                           VPolytope,
                                           Point,
                                           GraphOfConvexSetsOptions)
from pydrake.solvers import MosekSolver
from pydrake.planning import GcsTrajectoryOptimization

class SingleMazeGCSWorkspace:
    def __init__(self, cfg):
        obstacles = gcs_utils.create_test_box_env()
        bounds = np.array([[0, 5], [0, 5]])
        self.maze = MazeEnvironment(bounds, obstacles=obstacles, 
                                obstacle_padding=0.1)

        self.source = cfg.source
        self.num_trajectories = cfg.num_trajectories

        if cfg.task_id is not None:
            cfg.data_dir = f'{cfg.data_dir}/{cfg.task_id}'
        self.data_dir = cfg.data_dir
        self.zarr_name = "single_maze_gcs.zarr"

        # GCS settings
        self.max_rounded_paths = cfg.max_rounded_paths
        self.max_velocity = cfg.max_velocity
        self.continuity_order = cfg.continuity_order
        self.bezier_order = cfg.bezier_order

    def run(self):
        # Generate the random maze environments

        """
        Generates data and saves as zarr

        <data_dir>
        ├── single_maze_gcs.zarr
            ├── data
                ├── states
                ├── actions
            ├── meta
                ├── episode_ends
        ├── config.yaml
        """
        logging.getLogger('drake').setLevel(logging.WARNING)
        data = self.generate_data()
        
        # Save data as zarr
        

    
    def generate_data(self):
        """
        Generate data for a single maze and single source

        Returns an array of tuples (target, trajectory)
        """
        seed = int((datetime.now().timestamp() % 1) * 1e6)
        np.random.seed(seed)

        data = []
        pbar = \
            tqdm(total=self.num_trajectories, 
                position=0,
                miniters=30.0,
                desc='Trajectory generation')
        
        # Build graph
        start_time = time.time()
        gcs = GcsTrajectoryOptimization(self.maze.dim)
        free_space = gcs.AddRegions(self.maze.regions, self.bezier_order)
        source_region = gcs.AddRegions([Point(self.source)], 0)
        gcs.AddEdges(source_region, free_space)

        # Cost & Constraints
        gcs.AddVelocityBounds(np.array([-self.max_velocity, -self.max_velocity]),
                            np.array([self.max_velocity, self.max_velocity]))
        gcs.AddPathContinuityConstraints(self.continuity_order)
        gcs.AddTimeCost()
        gcs.AddPathLengthCost()
        time_to_build_graph = time.time() - start_time

        # Configure Options
        options = GraphOfConvexSetsOptions()
        options.max_rounded_paths = self.max_rounded_paths
        
        while len(data) < self.num_trajectories:
            start_time = time.time()
            goal = self.maze.sample_end_point()
            target_region = gcs.AddRegions([Point(goal)], 0)
            gcs.AddEdges(free_space, target)

            [traj, result] = gcs.SolvePath(source_region, target_region, options)
            end_time = time.time()

            # Rebuild GCS object if solve times become too slow
            factor = 1.5
            if end_time-start_time > factor * time_to_build_graph:
                gcs = GcsTrajectoryOptimization(self.maze.dim)
                free_space = gcs.AddRegions(self.maze.regions, self.bezier_order)
                source_region = gcs.AddRegions([Point(self.source)], 0)
                gcs.AddEdges(source_region, free_space)
                gcs.AddVelocityBounds(np.array([-self.max_velocity, -self.max_velocity]),
                                    np.array([self.max_velocity, self.max_velocity]))
                gcs.AddPathContinuityConstraints(self.continuity_order)
                gcs.AddTimeCost()
                gcs.AddPathLengthCost()
            
            if not result.is_success():
                continue

            # Successful trajectory => collect the data
            waypoints = gcs_utils.composite_trajectory_to_array(traj).transpose()
            data.append((goal, waypoints))
            pbar.update(1)
        
        pbar.close()
        return data