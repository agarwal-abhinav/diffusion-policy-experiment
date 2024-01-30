import pickle
import numpy as np
import zarr
import argparse
import os
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from diffusion_policy.common.replay_buffer import get_optimal_chunks

def main():
    """
    Converts data generated from MIT Supercloud into a single zarr
    The following is the expected structure of the data_dir
    
    data_dir
    |-- 0
    |   |-- maze_data.pkl
    |   |-- config.yaml
    |-- 1
    |   |-- maze_data.pkl
    |   |-- config.yaml
    ...
    
    The script will store the generated zarr file in data_dir.
    It will only store trajectories from the subdirectories [start_idx, end_idx)

    Usage:
    python data_generation/maze/convert_pkl_to_zarr.py --data_dir <path_to_data_dir> \
        --start_idx <start_idx> --end_idx <end_idx>
    
    """

    # parse data_dir from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    args = parser.parse_args()

    assert args.data_dir is not None
    data_dir = args.data_dir
    if args.start_idx is None:
        assert args.end_idx is None
        args.start_idx = 0
        args.end_idx = len(next(os.walk(args.data_dir))[1])
    else:
        assert args.end_idx is not None
        assert args.start_idx < args.end_idx

    config_path = os.path.join(args.data_dir, '0/config.yaml')
    with open(config_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Visualize data
    # dir_idx = 0
    # datapath = os.path.join(data_dir, str(dir_idx), 'maze_data.pkl')
    # data = pickle.load(open(datapath, 'rb'))
    # for maze_data in data:
    #     maze_env = maze_data['maze']
    #     for i in range(len(maze_data['trajectories'])):
    #         fig, ax = plt.subplots()
    #         def animate(idx):
    #             # apply the 62 by 62 crop as well
    #             plt.imshow(maze_data['imgs'][i][idx][1:-2, 1:-2, :])
    #         num_frames = len(maze_data['imgs'][i])
    #         dt = 0.1
    #         animation = FuncAnimation(fig, animate, frames=num_frames, 
    #                             interval=dt*1000, repeat=True)
    #         plt.show()
    # return

    # Compute size of arrays and episode ends
    episode_ends = []
    total_points = 0
    for dir_idx in tqdm(range(args.start_idx, args.end_idx)):
        datapath = os.path.join(data_dir, str(dir_idx), 'maze_data.pkl')
        data = pickle.load(open(datapath, 'rb'))

        for maze_data in data:
            trajectories = maze_data['trajectories']
            for i in range(len(trajectories)):
                trajectory = trajectories[i]
                total_points += trajectory.shape[0]
                episode_ends.append(total_points)
    episode_ends = np.array(episode_ends)
    print("Computed episode ends.")
    
    # Allocate memory for the arrays (these are VERY large)
    state = np.zeros((total_points, 2)).astype(np.float32)
    action = np.zeros((total_points, 2)).astype(np.float32)
    target = np.zeros((total_points, 2)).astype(np.float32)
    img = np.zeros((total_points, 
                    cfg["image_size"][0],
                    cfg["image_size"][1],
                    3)).astype(np.float32)
    print("Allocated memory for arrays.")

    # Set up progress bar
    num_pkl_files = args.end_idx - args.start_idx
    total_trajectories = num_pkl_files * \
        cfg['num_mazes_per_proc'] * cfg['num_trajectories_per_maze']
    pbar = tqdm(total=total_trajectories, position=0, leave=True)

    traj_idx = 0
    for dir_idx in range(args.start_idx, args.end_idx):
        datapath = os.path.join(data_dir, str(dir_idx), 'maze_data.pkl')
        data = pickle.load(open(datapath, 'rb'))
        
        for maze_data in data:
            targets = maze_data['targets']
            trajectories = maze_data['trajectories']
            imgs = maze_data['imgs']
            num_trajectories = len(trajectories)

            for i in range(num_trajectories):
                trajectory = trajectories[i]
                goal = targets[i]
                shifted_trajectory = \
                    np.concatenate([trajectory[1:, :], trajectory[-1:, :]], axis=0)
                
                traj_start_idx = episode_ends[traj_idx]-trajectory.shape[0]
                traj_end_idx = episode_ends[traj_idx]
                state[traj_start_idx:traj_end_idx, :] = trajectory
                action[traj_start_idx:traj_end_idx, :] = shifted_trajectory
                target[traj_start_idx:traj_end_idx, :] = np.array([goal]*trajectory.shape[0])
                img[traj_start_idx:traj_end_idx, :, :, :] = imgs[i]
                
                traj_idx += 1
                pbar.update(1)

    pbar.close()
    print("Concatenated data.")

    # Create zarr file and group structure
    if args.start_idx is None:
        zarr_path = os.path.join(args.data_dir, 'maze_image_dataset.zarr')
    else:
        zarr_path = os.path.join(args.data_dir, 
            f'maze_image_dataset_{args.start_idx}_{args.end_idx}.zarr')
    root = zarr.open_group(zarr_path, mode='w')
    data_dir = root.create_group('data')
    meta_dir = root.create_group('meta')

    # Chunk sizes optimized for read (not for supercloud storage, sorry admins)
    state_chunk_size = (1024, 2)
    action_chunk_size = (2048, 2)
    target_chunk_size = (1024, 2)
    image_chunk_size = (64, cfg["image_size"][0], cfg["image_size"][1], 3)

    # state_chunk_size = get_optimal_chunks(
    #     shape=state.shape, dtype=state.dtype)
    # target_chunk_size = get_optimal_chunks(
    #     shape=target.shape, dtype=target.dtype)
    # image_chunk_size = get_optimal_chunks(
    #     shape=img.shape, dtype=img.dtype)
    
    # Store data
    data_dir.create_dataset('state', data=state, chunks=state_chunk_size, dtype='f4')
    print("Stored state data. Chunk size: ", state_chunk_size)
    data_dir.create_dataset('action', data=action, chunks=action_chunk_size, dtype='f4')
    print("Stored action data. Chunk size: ", action_chunk_size)
    data_dir.create_dataset('target', data=target, chunks=target_chunk_size, dtype='f4')
    print("Stored target data. Chunk size: ", target_chunk_size)
    data_dir.create_dataset('img', data=img, chunks=image_chunk_size, dtype='f4')
    print("Stored img data. Chunk size: ", image_chunk_size)
    meta_dir.create_dataset('episode_ends', data=episode_ends)
    print("Stored episode_ends data. Chunk size: default", )
    print("All done.")

if __name__ == '__main__':
    main()