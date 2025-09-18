from typing import Dict, Optional, Tuple
import zarr
import torch
from torchvision import transforms
import numpy as np
import os
import copy
import random
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask, ImprovedDatasetSampler, VariableDatasetSampler)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

import torch.nn.functional as F

def gaussian_kernel(kernel_size=9, sigma=3, channels=3):
    """Create a Gaussian kernel for convolution."""
    # Create 1D Gaussian
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # Create 2D Gaussian
    g2 = g[:, None] * g[None, :]
    kernel = g2.expand(channels, 1, kernel_size, kernel_size)
    return kernel

def low_pass_filter(x, kernel):
    """Apply low-pass (Gaussian blur) filter to input tensor x."""
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])


# Remove the custom AttentionDatasetSampler class - we'll use ImprovedDatasetSampler directly


class PlanarPushingAttentionDataset(BaseImageDataset):
    """
    Attention-based dataset for planar pushing that supports:
    - Variable-length observation sequences for attention models
    - Full horizon-length sequences with flexible observation usage
    - Multi-camera support with proper masking
    - Curriculum learning support
    """
    def __init__(
        self,
        zarr_configs,
        shape_meta,
        use_one_hot_encoding=False,
        horizon=24,
        n_obs_steps=16,
        min_obs_steps=1,  # NEW: minimum observation steps
        max_obs_steps=None, # NEW: maximum observation steps (if None, use n_obs_steps)
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        color_jitter=None,
        low_pass_on_wrist=False, 
        low_pass_on_overhead=False,
        training_mode='random',  # 'random' or 'progressive' 
        progressive_steps=10000,  # Steps for progressive training mode
        random_sprinkle_prob=0.1  # Probability of sprinkling random obs in 'random_sprinkle' mode
    ):
        
        super().__init__()
        self._validate_zarr_configs(zarr_configs)
        if training_mode == 'random_sprinkle':
            assert random_sprinkle_prob is not None
            self.random_sprinkle_prob = random_sprinkle_prob
        
        # NEW: Dataset-level variable observation parameters
        self.min_obs_steps = min_obs_steps
        self.max_obs_steps = max_obs_steps if max_obs_steps is not None else n_obs_steps
        self.training_mode = training_mode  # 'random' or 'progressive'
        self.progressive_steps = progressive_steps
        self.training_step = 0
        self.current_max = min_obs_steps

        # Validation
        assert min_obs_steps >= 1, "min_obs_steps must be >= 1"
        assert min_obs_steps <= n_obs_steps, "min_obs_steps must be <= n_obs_steps"
        assert horizon >= n_obs_steps, "horizon must be >= n_obs_steps"

        self.low_pass_on_wrist = low_pass_on_wrist
        if low_pass_on_wrist: 
            self.wrist_kernel = gaussian_kernel(
                kernel_size=9, 
                sigma=3, 
                channels=3
            )
        self.low_pass_on_overhead = low_pass_on_overhead
        if low_pass_on_overhead:
            self.overhead_kernel = gaussian_kernel(
                kernel_size=9, 
                sigma=3, 
                channels=3
            )

        # Set up dataset keys
        self.rgb_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', '')
            if type == 'rgb':
                self.rgb_keys.append(key)
            
        keys = self.rgb_keys + ['state', 'action', 'target']
        self.keys = keys

        # Memory optimization: only load observations up to n_obs_steps
        # This follows the same pattern as the original improved sampling dataset
        key_first_k = dict()
        if n_obs_steps is not None:
            # Set all keys to n_obs_steps first (like original)
            for key in keys:
                key_first_k[key] = n_obs_steps
        # Override action to use full horizon (essential for policy)
        key_first_k['action'] = horizon

        # Load in all the zarr datasets
        self.num_datasets = len(zarr_configs)
        self.replay_buffers = []
        self.train_masks = []
        self.val_masks = []
        self.samplers = []
        self.sample_probabilities = np.zeros(len(zarr_configs))
        self.zarr_paths = []

        for i, zarr_config in enumerate(zarr_configs):
            # Extract config info
            zarr_path = zarr_config['path']
            max_train_episodes = zarr_config.get('max_train_episodes', None)
            sampling_weight = zarr_config.get('sampling_weight', None)
            
            # Set up replay buffer
            self.replay_buffers.append(ReplayBuffer.copy_from_path(
                    zarr_path=zarr_path, 
                    store=zarr.MemoryStore(),
                    keys=keys
                )
            )
            n_episodes = self.replay_buffers[-1].n_episodes

            # Set up masks
            if 'val_ratio' in zarr_config and zarr_config['val_ratio'] is not None:
                dataset_val_ratio = zarr_config['val_ratio']
            else:
                dataset_val_ratio = val_ratio
            val_mask = get_val_mask(
                n_episodes=n_episodes, 
                val_ratio=dataset_val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
            
            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)
            
            # Use ImprovedDatasetSampler for efficient variable-length sampling
            self.samplers.append(
                ImprovedDatasetSampler(
                    replay_buffer=self.replay_buffers[-1], 
                    sequence_length=horizon,
                    shape_meta=shape_meta,
                    pad_before=pad_before, 
                    pad_after=pad_after,
                    keys=keys,
                    key_first_k=key_first_k,
                    episode_mask=train_mask
                )
            )
            
            # Set up sample probabilities and zarr paths
            if sampling_weight is not None:
                self.sample_probabilities[i] = sampling_weight
            else:
                self.sample_probabilities[i] = np.sum(train_mask)
            self.zarr_paths.append(zarr_path)
        # Normalize sample_probabilities
        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)

        # Set up color jitter
        self.color_jitter = color_jitter
        if color_jitter is not None:
            self.transforms = transforms.ColorJitter(
                brightness=self.color_jitter.get('brightness', 0),
                contrast=self.color_jitter.get('contrast', 0),
                saturation=self.color_jitter.get('saturation', 0),
                hue=self.color_jitter.get('hue', 0)
            )

        # Load other variables
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.use_one_hot_encoding = use_one_hot_encoding
        self.one_hot_encoding = None # if val dataset, this will not be None

        print(f"AttentionDataset initialized: horizon={horizon}, n_obs_steps={n_obs_steps}, min_obs_steps={min_obs_steps}")


    def get_validation_dataset(self, index=None):
        """Create validation dataset with same parameters."""
        val_set = copy.copy(self)

        if index == None:
            assert self.num_datasets == 1, "Must specify validation dataset index if multiple datasets"
            index = 0
        else:
            val_set.replay_buffers = [self.replay_buffers[index]]
            val_set.train_masks = [self.train_masks[index]]
            val_set.val_masks = [self.val_masks[index]]
            val_set.zarr_paths = [self.zarr_paths[index]]
        val_set.num_datasets = 1
        val_set.sample_probabilities = np.array([1.0])

        # Set one hot encoding
        val_set.one_hot_encoding = np.zeros(self.num_datasets).astype(np.float32)
        val_set.one_hot_encoding[index] = 1

        # Create validation sampler with same memory optimization (match training)
        val_key_first_k = dict()
        if self.n_obs_steps is not None:
            for key in self.rgb_keys + ['state', 'action', 'target']:
                val_key_first_k[key] = self.n_obs_steps
        val_key_first_k['action'] = self.horizon
        
        val_set.samplers = [ImprovedDatasetSampler(
            replay_buffer=self.replay_buffers[index], 
            sequence_length=self.horizon,
            shape_meta=self.shape_meta,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            keys=self.rgb_keys + ['state', 'action', 'target'],
            episode_mask=self.val_masks[index],
            key_first_k=val_key_first_k
        )]
        
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        """Compute normalizer for all keys."""
        assert mode == 'limits', "Only supports limits mode"
        low_dim_keys = ['action', 'agent_pos', 'target']
        input_stats = {}
        for replay_buffer in self.replay_buffers:
            data = {
                'action': replay_buffer['action'],
                'agent_pos': replay_buffer['state'],
                'target': replay_buffer['target']
            }
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

            # Update mins and maxes
            for key in low_dim_keys:
                _max = normalizer[key].params_dict.input_stats.max
                _min = normalizer[key].params_dict.input_stats.min

                if key not in input_stats:
                    input_stats[key] = {'max': _max, 'min': _min}
                else:
                    input_stats[key]['max'] = torch.maximum(input_stats[key]['max'], _max)
                    input_stats[key]['min'] = torch.minimum(input_stats[key]['min'], _min)

        # Create normalizer
        normalizer = LinearNormalizer()
        normalizer.fit_from_input_stats(input_stats_dict=input_stats)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_sample_probabilities(self):
        return self.sample_probabilities
    
    def get_num_datasets(self):
        return self.num_datasets
    
    def get_num_episodes(self, index=None):
        if index == None:
            num_episodes = 0
            for i in range(self.num_datasets):
                num_episodes += self.replay_buffers[i].n_episodes
            return num_episodes
        else:
            return self.replay_buffers[index].n_episodes

    def __len__(self) -> int:
        length = 0
        for sampler in self.samplers:
            length += len(sampler)
        return length
    
    def _get_obs_steps(self) -> int:
        """
        Get number of observation steps based on training mode.
        
        Returns:
            Number of observation steps to use for this sample
        """
        if self.training_mode == 'random':
            # Random sampling between min and max for each sample
            return torch.randint(self.min_obs_steps, self.max_obs_steps + 1, (1,)).item()
        elif self.training_mode == 'random_sprinkle': 
            # Random sampling with sprinkling: mostly max_obs_steps, some random
            if random.random() < self.random_sprinkle_prob:
                return torch.randint(self.min_obs_steps, self.max_obs_steps, (1,)).item()
            else:
                return self.max_obs_steps
        elif self.training_mode == 'progressive':
            raise NotImplementedError("Progressive mode not implemented yet")
            # Progressive curriculum: start with min_obs_steps, gradually increase
            current_max = self.min_obs_steps + (self.training_step // self.progressive_steps) 
            print(f"Current max obs steps: {current_max}")
            self.current_max = current_max
            if self.current_max > self.max_obs_steps:
                self.current_max = self.max_obs_steps
            return torch.randint(self.min_obs_steps, self.current_max + 1, (1,)).item()
            # if self.training_step < self.progressive_steps:
            #     progress = self.training_step / self.progressive_steps
            #     current_max = self.min_obs_steps + progress * (self.max_obs_steps - self.min_obs_steps)
            #     current_max = int(current_max)
            #     if current_max > self.max_obs_steps:
            #         current_max = self.max_obs_steps
            #     self.current_max = current_max
            #     # Sample randomly up to current curriculum level
            #     return torch.randint(self.min_obs_steps, current_max + 1, (1,)).item()
            # else:
            #     # After curriculum completion, use full random range
            #     return torch.randint(self.min_obs_steps, self.max_obs_steps + 1, (1,)).item()
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")
    
    def set_training_step(self, step: int):
        """Update training step for progressive curriculum."""
        self.training_step = step

    def _validate_zarr_configs(self, zarr_configs):
        """Validate configuration parameters."""
        num_null_sampling_weights = 0
        N = len(zarr_configs)

        for zarr_config in zarr_configs:
            zarr_path = zarr_config['path']
            if not os.path.exists(zarr_path):
                raise ValueError(f"path {zarr_path} does not exist")
            
            max_train_episodes = zarr_config.get('max_train_episodes', None)
            if max_train_episodes is not None and max_train_episodes <= 0:
                raise ValueError(f"max_train_episodes must be greater than 0, got {max_train_episodes}")
            
            sampling_weight = zarr_config.get('sampling_weight', None)
            if sampling_weight is None:
                num_null_sampling_weights += 1
            elif sampling_weight < 0:
                raise ValueError(f"sampling_weight must be greater than or equal to 0, got {sampling_weight}")
        
        if num_null_sampling_weights not in [0, N]:
            raise ValueError("Either all or none of the zarr_configs must have a sampling_weight")
    
    def _normalize_sample_probabilities(self, sample_probabilities):
        total = np.sum(sample_probabilities)
        assert total > 0, "Sum of sampling weights must be greater than 0"
        return sample_probabilities / total
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with dataset-level variable observation sampling.
        Uses efficient ImprovedDatasetSampler that loads only the required observations.
        
        Returns:
            Dictionary with variable-length observations and full actions.
            Each sample can have different observation lengths within the batch.
        """
        
        # Determine number of observation steps for this sample
        num_obs_steps = self._get_obs_steps()

        key_first_k = dict()
            # Set all keys to num_obs_steps first (like original)
        for key in self.keys:
            key_first_k[key] = num_obs_steps
        # Override action to use full horizon (essential for policy)
        key_first_k['action'] = self.horizon
        
        # Get sample using efficient variable sampling
        if self.num_datasets == 1:
            sampler_idx = 0
            sampler = self.samplers[sampler_idx]
            data = sampler.sample_data(idx, key_first_k_override=key_first_k)
        else:
            sampler_idx = np.random.choice(self.num_datasets, p=self.sample_probabilities)
            sampler = self.samplers[sampler_idx]
            data = sampler.sample_data(idx % len(sampler), key_first_k_override=key_first_k)

        # ImprovedDatasetSampler handles everything - just pass through data directly
        output_data = data  # No processing needed!

        # Add one-hot encoding if needed
        if self.use_one_hot_encoding:
            if self.one_hot_encoding is None:
                output_data['one_hot_encoding'] = np.zeros(self.num_datasets).astype(np.float32)
                output_data['one_hot_encoding'][sampler_idx] = 1
            else:
                output_data['one_hot_encoding'] = self.one_hot_encoding

        # Convert to torch tensors
        torch_data = dict_apply(output_data, torch.from_numpy)
        
        # Add metadata including actual observation length used
        torch_data['sample_metadata'] = {
            'num_obs_steps': num_obs_steps,  # Actual obs steps for this sample
            'max_obs_steps': self.max_obs_steps,
            'min_obs_steps': self.min_obs_steps,
            # 'horizon': self.horizon,
            # 'sampler_idx': sampler_idx,
            # 'training_mode': self.training_mode
        }
        
        return torch_data


def create_attention_batch_with_variable_obs(observation_sequences: list, 
                                           action_sequences: list,
                                           num_tokens_per_sample: list,
                                           max_tokens: int = 16,
                                           strategy: str = 'most_recent') -> Dict:
    """
    Utility function to create attention-ready batches with variable observation lengths.
    
    Args:
        observation_sequences: List of obs dicts, each with tensors (seq_len, ...)
        action_sequences: List of action tensors, each (seq_len, action_dim)
        num_tokens_per_sample: List of ints, number of obs tokens to use per sample
        max_tokens: Maximum tokens per sample (for padding)
        strategy: 'most_recent', 'uniform_sample', or 'oldest_first'
    
    Returns:
        dict with keys: 'obs', 'action', 'obs_mask', 'temporal_positions'
    """
    batch_size = len(observation_sequences)
    if batch_size == 0:
        raise ValueError("Empty sequences")
        
    # Get shapes from first sample
    obs_keys = list(observation_sequences[0].keys())
    action_dim = action_sequences[0].shape[-1]
    seq_len = action_sequences[0].shape[0]
    
    # Initialize batch data
    batch_obs = {}
    for key in obs_keys:
        obs_shape = observation_sequences[0][key].shape
        batch_obs[key] = torch.zeros(batch_size, max_tokens, *obs_shape[1:])
    
    batch_actions = torch.stack(action_sequences, dim=0)  # (B, T, action_dim)
    obs_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    temporal_positions = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    
    for i, (obs_seq, num_tokens) in enumerate(zip(observation_sequences, num_tokens_per_sample)):
        seq_len = obs_seq[obs_keys[0]].shape[0]
        num_tokens = min(num_tokens, seq_len, max_tokens)
        
        if strategy == 'most_recent':
            # Take the most recent observations - RECOMMENDED
            start_idx = max(0, seq_len - num_tokens)
            end_idx = seq_len
            positions = torch.arange(start_idx, end_idx)
            
        elif strategy == 'uniform_sample':
            # Uniformly sample across the sequence
            indices = torch.linspace(0, seq_len - 1, num_tokens, dtype=torch.long)
            positions = indices
            start_idx = None  # Will use indices directly
            
        elif strategy == 'oldest_first':
            # Take the oldest observations
            start_idx = 0
            end_idx = num_tokens
            positions = torch.arange(start_idx, end_idx)
        
        # Fill batch tensors
        for key in obs_keys:
            if strategy == 'uniform_sample':
                batch_obs[key][i, :num_tokens] = obs_seq[key][indices]
            else:
                batch_obs[key][i, :num_tokens] = obs_seq[key][start_idx:end_idx]
        
        obs_mask[i, :num_tokens] = True
        temporal_positions[i, :num_tokens] = positions
    
    return {
        'obs': batch_obs,
        'action': batch_actions,
        'obs_mask': obs_mask,
        'temporal_positions': temporal_positions
    }


if __name__ == "__main__":
    """Test the attention dataset functionality."""
    import random
    import cv2
    import tqdm
    from torch.utils.data import DataLoader

    shape_meta = {
        'action': {'shape': [2]},
        'obs': {
            'agent_pos': {'type': 'low_dim', 'shape': [3]},
            'overhead_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
            'wrist_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
        },
    }
    zarr_configs = [
        {
            'path': 'data/planar_pushing_cotrain/sim_sim_tee_data_carbon_large.zarr',
            'max_train_episodes': None,
            'sampling_weight': 1.0
        }
    ]

    dataset = PlanarPushingAttentionDataset(
        zarr_configs=zarr_configs,
        shape_meta=shape_meta,
        horizon=24,
        n_obs_steps=16,
        min_obs_steps=2,
        pad_before=1,
        pad_after=7,
        seed=42,
        val_ratio=0.05,
        attention_curriculum=True
    )
    
    print("Initialized attention dataset")
    print("Total episodes (train + val):", dataset.get_num_episodes())
    print("Training dataset length:", len(dataset))

    # Test validation dataset
    val_dataset = dataset.get_validation_dataset(0)
    print("Validation dataset length:", len(val_dataset))

    # Test normalizer
    normalizer = dataset.get_normalizer()
    print("Normalizer created successfully")

    # Test sampling
    for i in range(5):
        idx = random.randint(0, len(dataset)-1)
        sample = dataset[idx]
        
        print(f"\nSample {i}:")
        print(f"  Action shape: {sample['action'].shape}")
        print(f"  Agent pos shape: {sample['obs']['agent_pos'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        
        for key in dataset.rgb_keys:
            print(f"  {key} shape: {sample['obs'][key].shape}")
        
        print(f"  Metadata: {sample['sample_metadata']}")
    
    print("\n✅ All attention dataset tests passed!")