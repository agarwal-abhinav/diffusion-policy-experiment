"""
Usage:
python maze_eval/initial_eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import zarr
import wandb
import json
from collections import deque
from torch.utils.data import DataLoader
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from maze_eval.utils.gcs_utils import *

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
# @click.option('-d', '--device', default='cuda:0')
@click.option('-d', '--device', default='cpu')
def main(checkpoint, output_dir, device):
    # if os.path.exists(output_dir):
    #     click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get normalizer
    dataset: BaseLowdimDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseLowdimDataset)
    normalizer = dataset.get_normalizer()

    # get policy from workspace
    policy = workspace.model
    policy.set_normalizer(normalizer)
    if cfg.training.use_ema:
        policy = workspace.ema_model
        policy.set_normalizer(normalizer)
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    # get initial starting and end positions
    # test_traj = zarr.open(cfg.task.dataset.zarr_path, mode='r')
    # source = test_traj['data']['state'][0]
    # target = test_traj['data']['state'][-1]

    # get environment
    regions = create_env()
    bounds = np.array([[0, 5], [0, 5]])

    # get parameters
    horizon = cfg.policy.horizon
    action_dim = cfg.policy.action_dim
    obs_dim = cfg.policy.obs_dim
    action_horizon = cfg.policy.n_action_steps
    obs_horizon = cfg.policy.n_obs_steps
    B = 1 # batch size is 1
    num_traj = 10

    # plot_environment(regions, source, target, np.array(test_traj['data']['state']).transpose())

    with torch.no_grad():
        for i in range(num_traj):
            done = False
            
            source = sample_collision_free_point(regions, bounds)
            target = np.array([2.0, 1.0])

            waypoints = [source]
            obs_deque = deque([torch.from_numpy(source).reshape(B,1,2)] * obs_horizon,
                            maxlen=obs_horizon)
            while len(waypoints) <= 300:
                obs = deque_to_dict(obs_deque)
                action_seq = policy.predict_action(obs)['action'][0]
                for action in action_seq:
                    waypoints.append(action.cpu().detach().numpy())
                    obs_deque.append(action.reshape(B,1,2))

                    if np.linalg.norm(action.cpu().detach().numpy() - target) < 0.1:
                        done = True
                        break
                
                if done:
                    break

            waypoints = np.array(waypoints).transpose()
            # for i in range(len(waypoints[0])):
                # plot_environment(regions, source, target, waypoints[:,:i+1])
            plot_environment(regions, source, target, waypoints)

def deque_to_dict(obs_deque: deque) -> dict[str, torch.Tensor]:
    obs_array = torch.cat(list(obs_deque), axis=1)
    return {'obs': obs_array}
    
if __name__ == '__main__':
    main()
