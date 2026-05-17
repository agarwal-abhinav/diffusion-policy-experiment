import numpy as np
### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])
def sample_box_no_rand_pose():
    cube_position = np.array([0.1,0.5, 0.05])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])
def sample_box_pose_large():
    x_range = [-0.2, 0.25]
    y_range = [0.28, 0.8]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    print("DEBUG: cube position", cube_position)
    return np.concatenate([cube_position, cube_quat])

def sample_box_rand_train_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_box_rand_test_pose():
    large_x_range = [-0.2, 0.25]
    large_y_range = [0.28, 0.8]
    small_x_range = [0.0, 0.2]
    small_y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    points = []
    
    while len(points) < 1:
        # Sample a point in the large box
        x = np.random.uniform(large_x_range[0], large_x_range[1])
        y = np.random.uniform(large_y_range[0], large_y_range[1])
        
        # Check if the point is inside the small box
        if not (small_x_range[0] <= x <= small_x_range[1] and small_y_range[0] <= y <= small_y_range[1]):
            points.append((x, y))

    cube_position = np.array([points[0][0], points[0][1], 0.05])
    cube_quat = np.array([1, 0, 0, 0])

    print("DEBUG: cube position", cube_position)
    return np.concatenate([cube_position, cube_quat])
    
def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

def sample_insertion_pose_large():
    # Peg
    x_range = [0, 0.25]
    y_range = [0.28, 0.8]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.25, -0.1]
    y_range = [0.3, 0.8]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])
    print("DEBUG: peg pose, socket pose", peg_pose, socket_pose)
    return peg_pose, socket_pose
