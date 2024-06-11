from typing import Dict
import zarr
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PlanarPushingDataset(BaseImageDataset):
    """
    Dataset for planar pushing that supports:
    - hybrid observations (images + end effector state)
    - multi cameras
    - TODO: cotraining with multiple datasets and loss scaling
    """
    def __init__(self,
            zarr_path,
            shape_meta,  
            horizon=1,
            n_obs_steps=None,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            ):
        
        super().__init__()

        self.rgb_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', '')
            if type == 'rgb':
                self.rgb_keys.append(key)
            
        keys = self.rgb_keys + ['state', 'action', 'target']

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=zarr_path, 
            store=zarr.MemoryStore(),
            keys=keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in keys:
                key_first_k[key] = n_obs_steps
        key_first_k['action'] = horizon

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'target': self.replay_buffer['target']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # image normalizer
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        
        # normalizer is a dict containing normalizers for all the keys
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        target = sample['target'][0].astype(np.float32)
        agent_pos = sample['state'].astype(np.float32)

        data = {
            'obs': {
                'agent_pos': agent_pos, # T_obs, 3
            },
            'target': target, # 3
            'action': sample['action'].astype(np.float32) # T, 2
        }

        # Add images to data
        for key in self.rgb_keys:
            data['obs'][key] = np.moveaxis(sample[key],-1,1)/255.0

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == "__main__":
    import random
    import cv2

    shape_meta = {
        'action': {'shape': [2]},
        'obs': {
            'agent_pos': {'type': 'low_dim', 'shape': [3]},
            'overhead_camera': {'type': 'rgb', 'shape': [3, 96, 96]},
            'overhead_camera_2': {'type': 'rgb', 'shape': [3, 96, 96]},
        },
    }

    n_obs_steps = 4

    dataset = PlanarPushingDataset(
        zarr_path='data/planar_pushing/test_multi_camera.zarr',
        shape_meta=shape_meta,
        horizon = 8,
        n_obs_steps = n_obs_steps,
        pad_before = 1,
        pad_after = 7,
        seed=42,
        val_ratio=0.05,
        max_train_episodes=None,
    )

    for _ in range(10):
        idx = random.randint(0, len(dataset)-1)
        sample = dataset[idx]
        states = sample['obs']['agent_pos']
        actions = sample['action']
        print(f"Sample states : {states}")
        print(f"Sample actions: {actions}")
        print()
        print("Press any key to continue. Ctrl+\\ to exit.\n")

        for key, attr in sample['obs'].items():
            if key == 'agent_pos':
                continue

            for i in range(len(attr)):
                image_array = attr[i].detach().numpy().transpose(1, 2, 0)

                # Convert the RGB array to BGR
                image_array[:,:,0], image_array[:,:,2] = image_array[:,:,2], image_array[:,:,0].copy()
                # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                # Display the image using OpenCV
                cv2.imshow(f'{key}_{i}', image_array)
                cv2.waitKey(0)  # Wait for a key press to close the image window
                cv2.destroyAllWindows()

        breakpoint()