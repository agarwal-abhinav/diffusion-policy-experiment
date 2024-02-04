import numpy as np
from tqdm import tqdm
from data_generation.motion_planners.tree import TreeNode, Tree
import kdtree # TODO: figure this out

def _euclidean_distance(a, b):
    """
    Returns the Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)

class BaseRRT:
    def __init__(self, source, max_step_size=0.1):
        self.RRT_tree = Tree(TreeNode(source))
        # used for nearest neighbor queries
        self.configuration_to_node = {
            tuple(source): self.RRT_tree.root
        }
        self.max_step_size = max_step_size
        self.kdtree = kdtree.create([source])

    def sample_free(self):
        """
        Returns a random point in the free space.
        """
        raise NotImplementedError()
    
    def is_free(self, q):
        """
        Returns True if the configuration q is in the free space.
        """
        raise NotImplementedError()

    def find_nearest(self, q, 
                     distance_metric=_euclidean_distance) -> TreeNode:
        """
        Returns the nearest node to configuration q in the tree.

        Args:
            q: a configuration in the configuration space
            distance_metric: a function that takes in two 
            configurations and returns the distance between them

        If no custom distance function is provided, the default
        this function will use a kd-tree to find the nearest node.

        If a custom distance function is provided, brute force
        is used instead of a kd-tree which runs in O(V) time.
        Using this method, the RRT algorithm runs in O(V^2) time.
        To improve runtime, override this method with something
        like a kd-tree.
        """
        nearest_node = None
        nearest_distance = float('inf')
        # kd tree
        if distance_metric is _euclidean_distance:
            kd_nearest, _ = self.kdtree.search_nn(q)
            nearest_distance = _euclidean_distance(kd_nearest.data, q)
            nearest_node = self.configuration_to_node[tuple(kd_nearest.data)]
        # brute force
        else:
            for vertex in self.configuration_to_node.values():
                distance = distance_metric(q, vertex.value)
                if distance < nearest_distance:
                    nearest_node = vertex
                    nearest_distance = distance
        return nearest_node, nearest_distance
    
    def steer(self, q, nearest_node, distance):
        """
        Returns a new configuration by moving from nearest_node to q
        in the direction of q, with a maximum distance of step_size.
        """
        step_size = min(self.max_step_size, distance)
        return nearest_node.value + \
            (q - nearest_node.value) * step_size / distance
    
    def add_vertex(self, q):
        """
        Adds a new vertex to the tree with value q.
        Returns new_node if success, None otherwise.
        """
        nearest_node, distance = self.find_nearest(q)
        q_new = self.steer(q, nearest_node, distance)

        # collision check
        if not self.is_free(q_new):
            return None
        if not self._obstacle_free(nearest_node.value, q_new):
            return None
        
        # collision-free, add new node to the tree
        new_node = TreeNode(q_new, nearest_node)
        self.RRT_tree.add_node(new_node, nearest_node)
        self.configuration_to_node[tuple(q_new)] = new_node
        self.kdtree.add(tuple(q_new))
        return new_node
    
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
        
    def find_path(self, q_goal):
        """
        Returns a path from the root to q_goal.
        If no path is found, return None
        """
        # check for collisions
        if not self.is_free(q_goal):
            return None
        nearest_node, distance = self.find_nearest(q_goal)
        if not self._obstacle_free(nearest_node.value, q_goal, 
                               n=int(distance // 0.1)):
            return None
        
        # passed collision checks: find path
        path = self.RRT_tree.find_path_to_root(nearest_node)
        path.append(q_goal)
        return path
    
    def shortcut_path(self, path, num_attempts=100):
        """
        Shortcuts the path using the straight line path between
        two configurations.
        """
        for _ in range(num_attempts):
            if len(path) < 3:
                return path
            # sample indices i and j st.
            # 0 <= i < j < len(path) and j - i > 1
            i = np.random.randint(0, len(path)-2)
            j = np.random.randint(i+2, len(path))
            num_points = int(_euclidean_distance(path[i], path[j]) // 0.1)
            if self._obstacle_free(path[i], path[j], num_points):
                path = path[:i+1] + path[j:]
        return path
    
    def visualize(self, path=None):
        """
        Visualizes the tree and the path.
        """
        raise NotImplementedError
    
    def _obstacle_free(self, q1, q2, n=3):
        """
        Returns True if the straight line path between q1 and q2 is obstacle-free.
        """
        for i in range(n):
            q_intermediate = q1 + (q2 - q1) * (i + 1) / (n+2)
            if not self.is_free(q_intermediate):
                return False
        return True
    