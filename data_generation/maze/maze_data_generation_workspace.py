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

import data_generation.maze.gcs_utils as gcs_utils
from data_generation.maze.maze_environment import MazeEnvironment
from data_generation.maze.maze_environment_generator import MazeEnvironmentGenerator
from pydrake.geometry.optimization import (HPolyhedron,
                                           VPolytope,
                                           Point,
                                           GraphOfConvexSetsOptions)
from pydrake.solvers import MosekSolver
from pydrake.planning import GcsTrajectoryOptimization

class MazeDataGenerationWorkspace:
    def __init__(self, maze_generator: MazeEnvironmentGenerator,
                  num_mazes_per_proc: int, 
                  num_trajectories_per_maze: int,
                  num_processes: int,
                  data_dir: str,
                  append_date_time: bool=True,
                  image_size: np.ndarray=np.ndarray([64, 64]),
                  # GCS settings
                  max_rounded_paths: int=3, # number of GCS rounded paths
                  max_velocity: float=1.0,
                  continuity_order: int=1,
                  bezier_order: int=3):

        self.maze_generator = maze_generator
        self.num_mazes_per_proc = num_mazes_per_proc
        self.num_trajectories_per_maze = num_trajectories_per_maze
        self.num_processes = num_processes
        
        now = datetime.now()
        self.data_dir = data_dir
        if append_date_time:
            self.data_dir = f'{data_dir}_{now.strftime("%d-%m-%Y_%H:%M:%S")}'

        # GCS settings
        self.max_rounded_paths = max_rounded_paths
        self.max_velocity = max_velocity
        self.continuity_order = continuity_order
        self.bezier_order = bezier_order

    def run(self):
        # Generate the random maze environments

        """
        data is an array of dictionaries
        Each dictionary has the following keys:
        - maze: the maze environment object
        - sources: a list of start points
        - targets: a list of end points
        - trajectories: a list of trajectories
        - imgs: a list of images corresponding to the trajectories
        """
        logging.getLogger('drake').setLevel(logging.WARNING)

        data = []
        # parallel data collection
        with Pool(self.num_processes) as p:
            pooled_data = p.starmap(self.generate_data, 
                                    [() for _ in range(self.num_processes)])
        
        for d in pooled_data:
            data.extend(d)
                
        # Save with pickles
        # TODO: save with Zarr
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=False)
        with open(f'{self.data_dir}/maze_data.pkl', 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    
    def generate_data(self):
        # Generate seed for thread
        seed = int((datetime.now().timestamp() % 1) * 1e6)
        np.random.seed(seed)

        mazes = [self.maze_generator.generate_maze_environment() \
                 for i in range(self.num_mazes_per_proc)]
        data = []

        pbar = \
            tqdm(total=self.num_mazes_per_proc * self.num_trajectories_per_maze, 
                position=0)
        for maze in mazes:
            dim = maze.dim

            # Build graph
            start_time = time.time()
            gcs = GcsTrajectoryOptimization(dim)
            free_space = gcs.AddRegions(maze.regions, self.bezier_order)

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

            # Generate data
            maze_data = {'maze': maze,
                         'sources': [],
                         'targets': [],
                         'trajectories': [],
                         'imgs': []
            }

            while len(maze_data['trajectories']) < self.num_trajectories_per_maze:
                start_time = time.time()
                start = maze.sample_start_point()
                goal = maze.sample_end_point()
                source = gcs.AddRegions([Point(start)], 0)
                target = gcs.AddRegions([Point(goal)], 0)
                gcs.AddEdges(source, free_space)
                gcs.AddEdges(free_space, target)

                [traj, result] = gcs.SolvePath(source, target, options)
                end_time = time.time()

                # Rebuild GCS object if solve times become too slow
                factor = 1.5
                if end_time-start_time > factor * time_to_build_graph:
                    gcs = GcsTrajectoryOptimization(dim)
                    free_space = gcs.AddRegions(maze.regions, self.bezier_order)
                    gcs.AddVelocityBounds(np.array([-self.max_velocity, -self.max_velocity]),
                                        np.array([self.max_velocity, self.max_velocity]))
                    gcs.AddPathContinuityConstraints(self.continuity_order)
                    gcs.AddTimeCost()
                    gcs.AddPathLengthCost()
                
                if not result.is_success():
                    continue

                # Successful trajectory => collect the data
                maze_data['sources'].append(start)
                maze_data['targets'].append(goal)
                maze_data['trajectories'].append(
                    gcs_utils.composite_trajectory_to_array(traj).transpose()
                )
                # Generate corresponding images
                imgs = []
                for waypoint in maze_data['trajectories'][-1]:
                    imgs.append(maze.to_img(waypoint))
                maze_data['imgs'].append(np.array(imgs))
                
                pbar.update(1)
            
            data.append(maze_data)
        
        pbar.close()
        return data
    
if __name__ == "__main__":
    maze_generator = MazeEnvironmentGenerator(
        min_num_obstacles=10,
        max_num_obstacles=15,
        min_obstacle_width=0.2,
        max_obstacle_width=1.5,
        min_obstacle_height=0.2,
        max_obstacle_height=1.5,
        border_padding=0.3,
        bounds=np.array([[0,5.0], [0, 5.0]]),
        non_overlapping_centers=True
    )
    maze_data_generation_workspace = MazeDataGenerationWorkspace(
        maze_generator=maze_generator,
        num_mazes_per_proc=5,
        num_trajectories_per_maze=2,
        num_processes=1,
        data_dir='data_generation/maze_data',
        max_rounded_paths=10,
        max_velocity=1.0,
        continuity_order=1,
        bezier_order=3
    )

    maze_data_generation_workspace.run()
    data_file = f'{maze_data_generation_workspace.data_dir}/maze_data.pkl'
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    for i in range(5):
        maze_data = data[i]
        maze = maze_data['maze']
        for j in range(2):
            start = maze_data['sources'][j]
            end = maze_data['targets'][j]
            waypoints = maze_data['trajectories'][j]

            maze.plot_trajectory(start, end, waypoints)
            waypoint = waypoints[0]
            np_img = maze.to_img(position=waypoint)
            newfig, ax = plt.subplots()
            ax.imshow(np_img)
            plt.show()