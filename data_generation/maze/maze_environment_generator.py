import numpy as np

from typing import List
from data_generation.maze.maze_environment import MazeEnvironment
from pydrake.geometry.optimization import (HPolyhedron,
                                           VPolytope)

class MazeEnvironmentGenerator:
    """
    Class that generates MazeEnvironments 
    
    Note: This class generates mazes that satisfy the assumptions
    of the MazeEnvironment class.
    """

    def __init__(self, 
                 min_num_obstacles: int,
                 max_num_obstacles: int,
                 min_obstacle_width: float,
                 max_obstacle_width: float,
                 min_obstacle_height: float,
                 max_obstacle_height: float,
                 padding: float, # padding around edges
                 bounds: np.ndarray, # bounds of the maze
                 non_overlapping_centers: bool = True):
        
        self.min_num_obstacles = min_num_obstacles
        self.max_num_obstacles = max_num_obstacles
        self.min_obstacle_width = min_obstacle_width
        self.max_obstacle_width = max_obstacle_width
        self.min_obstacle_height = min_obstacle_height
        self.max_obstacle_height = max_obstacle_height
        self.padding = padding
        self.bounds = bounds
        self.padded_bounds = bounds + padding*np.array([[1, -1], [1, -1]])
        self.non_overlapping_centers = non_overlapping_centers

    def generate_obstacle(self, existing_obstacles: List[HPolyhedron]=None) -> HPolyhedron:
        # sample a center that is not in collision
        # this ensures a better spread of obstacles
        center_x, center_y = None, None
        while True:
            center_x = np.random.uniform(
                self.padded_bounds[0,0] + self.min_obstacle_width/2,
                self.padded_bounds[0,1] - self.min_obstacle_width/2
            )
            center_y = np.random.uniform(
                self.padded_bounds[0,0] + self.min_obstacle_height/2,
                self.padded_bounds[0,1] - self.min_obstacle_height/2
            )
            break_condition = (existing_obstacles == None) or \
                (not self.in_collision(np.array([center_x, center_y]), existing_obstacles))
            if break_condition:
                break

        width = np.random.uniform(
            self.min_obstacle_width,
            self.max_obstacle_width
        )
        height = np.random.uniform(
            self.min_obstacle_height,
            self.max_obstacle_height
        )

        left = max(center_x - width/2, self.padded_bounds[0,0])
        right = min(center_x + width/2, self.padded_bounds[0,1])
        bot = max(center_y - height/2, self.padded_bounds[1,0])
        top = min(center_y + height/2, self.padded_bounds[1,1])

        vertices = np.round(np.array(
            [[left, right, left, right],
             [bot, bot, top, top]]), 1
        )
        return HPolyhedron(VPolytope(vertices))

    def generate_obstacle_list(self) -> List[HPolyhedron]:
        num_obstacles = np.random.randint(
            self.min_num_obstacles,
            self.max_num_obstacles
        )
        if self.non_overlapping_centers:
            obstacles = []
            for i in range(num_obstacles):
                obstacles.append(self.generate_obstacle(obstacles))
            return obstacles
        else:
            return [self.generate_obstacle() for _ in range(num_obstacles)]
        

    def generate_maze_environment(self) -> MazeEnvironment:
        obstacles = self.generate_obstacle_list()
        return MazeEnvironment(bounds=self.bounds, obstacles=obstacles)
    
    """ Helper Functions """
    def compute_obstacle_center(self, vpolytope: VPolytope) -> np.ndarray:
        return np.mean(vpolytope.vertices(), axis=0)
    
    def in_collision(self, x: np.ndarray, obstacles: List[HPolyhedron]) -> bool:
        for obstacle in obstacles:
            if np.all(obstacle.A() @ x <= obstacle.b()):
                return True
        return False

if __name__ == '__main__':
    """ Test MazeEnvironmentGenerator """
    generator = MazeEnvironmentGenerator(
        min_num_obstacles=10,
        max_num_obstacles=15,
        min_obstacle_width=0.2,
        max_obstacle_width=1.5,
        min_obstacle_height=0.2,
        max_obstacle_height=1.5,
        padding = 0.3,
        bounds=np.array([[0, 5], [0, 5]])
    )
    for i in range(10):
        env = generator.generate_maze_environment()
        env.plot_environment()