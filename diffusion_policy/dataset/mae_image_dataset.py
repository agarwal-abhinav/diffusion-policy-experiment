"""
Frame-level image dataset for MAE pre-training.

Iterates over all individual frames across zarr replay buffers and camera
streams, returning single cropped+normalized images. Supports train/val
split at the episode level.
"""

import copy
import numpy as np
import torch
from torch.utils.data import Dataset
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask, downsample_mask
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer


class MAEImageDataset(Dataset):
    def __init__(
        self,
        zarr_configs: list,
        rgb_keys: list,
        crop_shape=(112, 112),
        img_shape=(3, 128, 128),
        imagenet_norm: bool = True,
        val_ratio: float = 0.05,
        seed: int = 42,
        max_train_episodes: int = None,
        # private — set by get_validation_dataset()
        _is_val: bool = False,
        _frame_indices: list = None,
        _replay_buffers: list = None,
    ):
        super().__init__()
        self.rgb_keys = list(rgb_keys)
        self.crop_shape = tuple(crop_shape)
        self.imagenet_norm = imagenet_norm
        self._is_val = _is_val

        # ImageNet normalization constants
        if imagenet_norm:
            self.register_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            self.register_std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)

        # CropRandomizer
        ch, cw = crop_shape
        c_in = img_shape[0]
        h_in, w_in = img_shape[1], img_shape[2]
        self.crop_randomizer = None
        if ch < h_in and cw < w_in:
            self.crop_randomizer = CropRandomizer(
                input_shape=(c_in, h_in, w_in),
                crop_height=ch,
                crop_width=cw,
                num_crops=1,
                pos_enc=False,
            )

        if _frame_indices is not None:
            # Pre-built (from create_train_val)
            self.replay_buffers = _replay_buffers
            self.frame_indices = _frame_indices
            print(f"MAEImageDataset: {len(self.frame_indices)} frames "
                  f"({'val' if _is_val else 'train'}, "
                  f"{len(self.replay_buffers)} zarr files, "
                  f"{len(self.rgb_keys)} cameras)")
            return

        # Load replay buffers and build frame index
        if _replay_buffers is not None:
            self.replay_buffers = _replay_buffers
        else:
            self.replay_buffers = []
        self.frame_indices = []  # list of (buf_idx, camera_key, frame_idx)

        for i, zarr_config in enumerate(zarr_configs):
            zarr_path = zarr_config['path']
            dataset_val_ratio = zarr_config.get('val_ratio', val_ratio)
            dataset_max_train = zarr_config.get('max_train_episodes', max_train_episodes)

            if _replay_buffers is None:
                replay_buffer = ReplayBuffer.copy_from_path(
                    zarr_path=zarr_path,
                    store=zarr.MemoryStore(),
                    keys=list(rgb_keys),
                )
                self.replay_buffers.append(replay_buffer)
            buf_idx = i
            replay_buffer = self.replay_buffers[buf_idx]

            n_episodes = replay_buffer.n_episodes
            episode_ends = replay_buffer.episode_ends[:]

            # Episode-level train/val split
            val_mask = get_val_mask(
                n_episodes=n_episodes,
                val_ratio=dataset_val_ratio,
                seed=seed)
            train_mask = ~val_mask
            train_mask = downsample_mask(
                mask=train_mask,
                max_n=dataset_max_train,
                seed=seed)

            # Pick the right mask
            episode_mask = val_mask if _is_val else train_mask

            # Expand episode mask to frame indices
            for ep_idx in range(n_episodes):
                if not episode_mask[ep_idx]:
                    continue
                start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
                end = int(episode_ends[ep_idx])
                for frame_idx in range(start, end):
                    for key in self.rgb_keys:
                        self.frame_indices.append((buf_idx, key, frame_idx))

        print(f"MAEImageDataset: {len(self.frame_indices)} frames "
              f"({'val' if _is_val else 'train'}, "
              f"{len(self.replay_buffers)} zarr files, "
              f"{len(self.rgb_keys)} cameras)")

    def get_validation_dataset(self):
        """Create a validation dataset from the same zarr configs."""
        # Rebuild with _is_val=True (same zarr configs, different mask)
        val_ds = MAEImageDataset.__new__(MAEImageDataset)
        Dataset.__init__(val_ds)
        val_ds.rgb_keys = self.rgb_keys
        val_ds.crop_shape = self.crop_shape
        val_ds.imagenet_norm = self.imagenet_norm
        val_ds._is_val = True
        if self.imagenet_norm:
            val_ds.register_mean = self.register_mean
            val_ds.register_std = self.register_std
        val_ds.crop_randomizer = copy.deepcopy(self.crop_randomizer)
        if val_ds.crop_randomizer is not None:
            val_ds.crop_randomizer.eval()
        val_ds.replay_buffers = self.replay_buffers  # share buffers (read-only)

        # Rebuild frame indices for val episodes
        # We need the original zarr_configs to re-derive the val mask,
        # but we can just re-instantiate
        # Actually, simpler: store zarr_configs and rebuild
        # But we don't have them... Let's use a different approach.
        # We'll store _zarr_configs on self during __init__
        raise NotImplementedError("Use the class constructor with _is_val=True instead")

    @classmethod
    def create_train_val(cls, zarr_configs, rgb_keys, crop_shape=(112, 112),
                         img_shape=(3, 128, 128), imagenet_norm=True,
                         val_ratio=0.05, seed=42, max_train_episodes=None):
        """Create both train and val datasets, sharing replay buffers."""
        train_ds = cls(
            zarr_configs=zarr_configs, rgb_keys=rgb_keys,
            crop_shape=crop_shape, img_shape=img_shape,
            imagenet_norm=imagenet_norm, val_ratio=val_ratio,
            seed=seed, max_train_episodes=max_train_episodes,
            _is_val=False)
        # Reuse loaded replay buffers for val (no re-loading from disk)
        val_ds = cls(
            zarr_configs=zarr_configs, rgb_keys=rgb_keys,
            crop_shape=crop_shape, img_shape=img_shape,
            imagenet_norm=imagenet_norm, val_ratio=val_ratio,
            seed=seed, max_train_episodes=max_train_episodes,
            _is_val=True,
            _replay_buffers=train_ds.replay_buffers,
        )
        return train_ds, val_ds

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        buf_idx, key, frame_idx = self.frame_indices[idx]
        replay_buffer = self.replay_buffers[buf_idx]

        # Load single frame: (H, W, C) uint8
        img = replay_buffer[key][frame_idx]
        img = torch.from_numpy(img.copy()).float()
        img = img.permute(2, 0, 1) / 255.0  # (C, H, W) float [0, 1]

        # Apply crop
        if self.crop_randomizer is not None:
            # CropRandomizer expects (B, C, H, W) and returns (B, C, H, W)
            img = img.unsqueeze(0)
            img = self.crop_randomizer(img)
            img = img.squeeze(0)

        # ImageNet normalization
        if self.imagenet_norm:
            img = (img - self.register_mean) / self.register_std

        return img
