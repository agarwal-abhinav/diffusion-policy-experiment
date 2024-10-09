import numpy as np
import hydra
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import dill
import pathlib
import os

from diffusion_policy.dataset.planar_pushing_dataset import PlanarPushingDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset

MISSING_CHECKPOINTS = []

def save_actions_and_embeddings(checkpoint_dir: str, save_dir: str):
    # Create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        print(f"Skipping {save_dir} as it already exists")
        return

    # Load model
    checkpoint = pathlib.Path(f"{checkpoint_dir}/checkpoints/latest.ckpt")
    if not os.path.exists(checkpoint):
        print(f"Skipping {checkpoint} as it does not exist")
        MISSING_CHECKPOINTS.append(checkpoint)
        return
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # Get workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get normalizer
    normalizer_path = checkpoint.parent.parent.joinpath("normalizer.pt")
    normalizer = torch.load(normalizer_path)

    # Get policy from workspace
    policy = workspace.model
    policy.set_normalizer(normalizer)
    if cfg.training.use_ema:
        policy = workspace.ema_model
        policy.set_normalizer(normalizer)
    device = torch.device("cuda")
    policy.to(device)
    policy.eval()

    # Get datasets
    zarr_configs = cfg.task.dataset.zarr_configs
    dataset_config = cfg.task.dataset
    datasets = {}
    for zarr_config in zarr_configs:
        dataset_config['zarr_configs'] = [zarr_config]
        if 'real_world_tee_data.zarr' in zarr_config['path']:
            datasets['real'] = hydra.utils.instantiate(dataset_config)
        else:
            datasets['sim'] = hydra.utils.instantiate(dataset_config)
    
    # Get dataloaders
    batch_size = 128
    real_dataloader = DataLoader(
        datasets['real'],
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False
    )
    sim_dataloader = DataLoader(
        datasets['sim'],
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False
    )

    # Compute embeddings
    num_real = len(datasets['real'])
    real_actions = torch.zeros((num_real, 16, 2))
    real_embeddings = torch.zeros((num_real, 262))
    with torch.no_grad():
        with tqdm(real_dataloader, desc=f"Real data") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))            
                actions = batch['action']
                embeddings = policy.compute_obs_embedding(batch)

                start_idx = batch_idx * batch_size
                end_idx = start_idx + actions.shape[0]
                real_actions[start_idx:end_idx] = actions
                real_embeddings[start_idx:end_idx] = embeddings

    # Save embeddings
    real_actions = real_actions.cpu().numpy()
    real_embeddings = real_embeddings.cpu().numpy()
    np.save(f"{save_dir}/real_actions.npy", real_actions)
    np.save(f"{save_dir}/real_embeddings.npy", real_embeddings)
    del real_actions
    del real_embeddings

    num_sim = len(datasets['sim'])
    sim_actions = torch.zeros((num_sim, 16, 2))
    sim_embeddings = torch.zeros((num_sim, 262))
    with torch.no_grad():
        with tqdm(sim_dataloader, desc=f"Sim data") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))            
                actions = batch['action']
                embeddings = policy.compute_obs_embedding(batch)

                start_idx = batch_idx * batch_size
                end_idx = start_idx + actions.shape[0]
                sim_actions[start_idx:end_idx] = actions
                sim_embeddings[start_idx:end_idx] = embeddings

    # Save embeddings
    sim_actions = sim_actions.cpu().numpy()
    sim_embeddings = sim_embeddings.cpu().numpy()
    np.save(f"{save_dir}/sim_actions.npy", sim_actions)
    np.save(f"{save_dir}/sim_embeddings.npy", sim_embeddings)
    del sim_actions
    del sim_embeddings    

def main():
    exclude_keywords = ["bugged", "test", "real_only", "misc"]
    for experiment in os.listdir("data/outputs/cotrain"):
        if any(keyword in experiment for keyword in exclude_keywords):
            continue
        experiment_dir = f"data/outputs/cotrain/{experiment}"
        if os.path.isdir(experiment_dir):
            for policy in os.listdir(experiment_dir):
                policy_dir = f"{experiment_dir}/{policy}"
                if os.path.isdir(policy_dir):
                    print(f"Saving actions and embeddings for {policy_dir}")
                    save_actions_and_embeddings(
                        f"data/outputs/cotrain/{experiment}/{policy}",
                        f"data/experiments/cotrain/{experiment}/{policy}"
                    )
    print(f"Missing checkpoints: {MISSING_CHECKPOINTS}")

if __name__ == '__main__':
    main()