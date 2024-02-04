import numpy as np
import math
from tqdm import tqdm
from data_generation.motion_planners.common import (
    TreeNode, Tree, _euclidean_distance, KDTreePayload
)
from data_generation.motion_planners.base_RRT import BaseRRT
import kdtree

class BaseRRTStar(BaseRRT):
    def __init__(self, source, max_step_size=0.1):
        super().__init__(source, max_step_size)
        self.k_RRT = 6 # any number greater than 2e will work

    def k_nearest_neighbors(self, q, k, 
                            distance_metric=_euclidean_distance):
        if distance_metric is _euclidean_distance:
            knn = self.kdtree.search_knn(q, k)
            distances = [distance_metric(x.data.q, q) for x in knn]
            return [(knn[i].data.data, distances[i]) for i in range(len(knn))]
        else:
            # Brute force solution
            raise NotImplementedError()

    def add_vertex(self, q):
        """
        Adds a new vertex to the tree with value q
        and performs rewiring.
        Returns new_node if success, None otherwise.
        """
        nearest_node, distance = self.find_nearest(q)
        q_new = self.steer(q, nearest_node, distance)

        # Collision check
        if not self.is_free(q_new):
            return None
        if not self._obstacle_free(nearest_node.value, q_new):
            return None
        
        # Collision-free, add new node to the tree
        k = self.k_RRT * math.log(len(self.vertices))
        nearest_neighbors = self.k_nearest_neighbors(q_new, k)
        # Connect along a minimum-cost path
        for node, distance in nearest_neighbors:
            # TODO:

    def _rewire_tree(self, q_new, nearest_neighbors):
        pass
    
    def sample_and_add_vertex(self):
        """
        Samples a new configuration and adds it to the tree.
        Returns True if success, False otherwise.
        """
        new_node = None
        while new_node is None:
            q = self.sample_free()
            new_node = self.add_vertex(q)
        return new_node

    def grow(self, N: int):
        """
        Grows the RRT tree.
        """
        for _ in tqdm(range(N), desc='Growing RRT'):
            self.sample_and_add_vertex()

    def grow_to_goal(self, q_goal,
                     distance_metric=_euclidean_distance,
                     num_shortcut_attempts: int=0):
        # needs to consider k nearest neighbors
        pass

    def find_path(self, q_goal, num_shortcut_attempts: int=0):
        # needs to consider k nearest neighbors
        pass