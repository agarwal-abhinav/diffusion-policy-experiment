import pickle
import numpy as np
import zarr
import argparse
import os
import yaml

from tqdm import tqdm

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
    
    The script will store the generated zarr file in data_dir

    Usage:
    python convert_pkl_to_zarr.py --data_dir <path_to_data_dir>
    """

    # parse data_dir from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    args = parser.parse_args()
    assert args.data_dir is not None
    config_path = os.path.join(args.data_dir, '0/config.yaml')

    state = []
    action = []
    target = []
    img = []
    episode_ends = []
    current_end = 0
    
    # Set up progress bar
    num_pkl_files = len(next(os.walk(args.data_dir))[1])
    with open(config_path, 'r') as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    total_trajectories = num_pkl_files * \
        cfg['num_mazes_per_proc'] * cfg['num_trajectories_per_maze']
    pbar = tqdm(total=total_trajectories, position=0, leave=True)

    for root, _, files in os.walk(args.data_dir):
        # load data from file
        if 'maze_data.pkl' not in files:
            continue

        datapath = os.path.join(root, 'maze_data.pkl')
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
                state.append(trajectory)
                action.append(shifted_trajectory)
                target.append(np.array([goal]*trajectory.shape[0]))
                img.append(imgs[i])

                current_end += trajectory.shape[0]
                episode_ends.append(current_end)
                
                pbar.update(1)

    pbar.close()
    
    # Convert data to numpy arrays
    state = np.concatenate(state, axis=0).astype(np.float32)
    action = np.concatenate(action, axis=0).astype(np.float32)
    target = np.concatenate(target, axis=0).astype(np.float32)
    img = np.concatenate(img, axis=0).astype(np.float32)
    episode_ends = np.array(episode_ends)
    print("Concatenated data.")

    # Create zarr file and group structure
    zarr_path = os.path.join(args.data_dir, 'maze_image_dataset_4000.zarr')
    root = zarr.open_group(zarr_path, mode='w')
    data_dir = root.create_group('data')
    meta_dir = root.create_group('meta')

    # Chunk sizes optimized for ~1-10MB chunks after compression
    state_chunk_size = (1000000, 2)
    target_chunk_size = (4000000, 2) # easier to compress
    image_chunk_size = (10000, 64, 64, 3)
    
    # Store data
    data_dir.create_dataset('state', data=state, chunks=state_chunk_size, dtype='f4')
    print("Stored state data.")
    data_dir.create_dataset('action', data=action, chunks=state_chunk_size, dtype='f4')
    print("Stored action data.")
    data_dir.create_dataset('target', data=target, chunks=target_chunk_size, dtype='f4')
    print("Stored target data.")
    data_dir.create_dataset('img', data=img, chunks=image_chunk_size, dtype='f4')
    print("Stored img data.")
    meta_dir.create_dataset('episode_ends', data=episode_ends)
    print("Stored episode_ends data. All done.")

if __name__ == '__main__':
    main()