#!/usr/bin/env python3

import torch
import numpy as np
from pathlib import Path
from diffusion_policy.dataset.planar_pushing_attention_dataset import PlanarPushingAttentionDataset

def debug_sampler():
    """Debug the sampler to see what's actually happening"""
    
    zarr_path = Path("data/planar_pushing_cotrain/sim_sim_tee_data_carbon_large.zarr")
    if not zarr_path.exists():
        print(f"Dataset not found")
        return
        
    print("🔍 DEBUGGING SAMPLER BEHAVIOR")
    print("=" * 40)
    
    shape_meta = {
        'action': {'shape': [2]},
        'obs': {
            'agent_pos': {'shape': [3], 'type': 'low_dim'},
            'overhead_camera': {'shape': [3, 128, 128], 'type': 'rgb'},
            'wrist_camera': {'shape': [3, 128, 128], 'type': 'rgb'}
        }
    }
    
    dataset = PlanarPushingAttentionDataset(
        zarr_configs=[{
            'path': str(zarr_path),
            'max_train_episodes': 5,
            'sampling_weight': 1.0
        }],
        shape_meta=shape_meta,
        horizon=16,                  
        n_obs_steps=8,              
        min_obs_steps=2,            
        training_mode='random',      
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.05,
    )
    
    print("Testing dataset __getitem__ method directly:")
    
    # Let's manually trace what happens
    for test_idx in range(3):
        print(f"\n--- Sample {test_idx} ---")
        
        # Check what _get_obs_steps returns
        num_obs_steps = dataset._get_obs_steps()
        print(f"1. _get_obs_steps() returned: {num_obs_steps}")
        
        # Check key_first_k construction  
        key_first_k = dict()
        for key in dataset.keys:
            key_first_k[key] = num_obs_steps
        key_first_k['action'] = dataset.horizon
        print(f"2. key_first_k created: {key_first_k}")
        
        # Check what sampler.sample_data returns
        if dataset.num_datasets == 1:
            sampler_idx = 0
            sampler = dataset.samplers[sampler_idx]
            data = sampler.sample_data(test_idx, key_first_k_override=key_first_k)
        
        print(f"3. Sampler returned data keys: {list(data.keys())}")
        if 'obs' in data:
            print(f"4. Obs keys: {list(data['obs'].keys())}")
            for key, value in data['obs'].items():
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                
                # Check for NaN
                if key == 'agent_pos':
                    nan_mask = torch.tensor(np.isnan(value))
                    if torch.any(nan_mask):
                        nan_per_timestep = nan_mask.view(value.shape[0], -1).any(dim=1)
                        first_nan_idx = torch.where(nan_per_timestep)[0]
                        if len(first_nan_idx) > 0:
                            print(f"   First NaN at: {first_nan_idx[0].item()}")
                        else:
                            print(f"   No clear NaN boundary found")
                    else:
                        print(f"   No NaN found")
                        
                    # Check actual values
                    print(f"   First few values: {value[:min(6, len(value))].flatten()}")
                    print(f"   Last few values: {value[-min(4, len(value)):].flatten()}")
        
        print(f"5. Action shape: {data['action'].shape}")
        
        if test_idx >= 2:
            break

if __name__ == "__main__":
    debug_sampler()