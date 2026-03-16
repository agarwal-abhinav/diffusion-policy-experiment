# Diffusion Policy Experiment Repository

## Overview
Training pipeline for diffusion-based robot manipulation policies. Uses Hydra for config management, wandb for logging, zarr for data storage.

## Training Entry Point
```bash
python train.py --config-dir=config/<path> --config-name=<config_name>
```
`train.py` uses Hydra to instantiate a workspace class (via `_target_`) and calls `workspace.run()`.

## Key Architecture

### Directory Structure
- `diffusion_policy/workspace/` — Training loop classes (23 variants)
- `diffusion_policy/policy/` — Policy networks (30 variants)
- `diffusion_policy/dataset/` — Dataset loaders (25+ variants)
- `diffusion_policy/model/` — Neural network components (UNet, Transformer, Vision encoders, EMA)
- `diffusion_policy/common/` — Utilities (replay_buffer, sampler, normalizers, checkpoint)
- `diffusion_policy/env/` — Environments (block_pushing, kitchen, pusht, robomimic)
- `diffusion_policy/env_runner/` — Evaluation runners
- `config/` — Hydra YAML configs organized by task/experiment

### Primary Pipeline (IROS long-context planar pushing)
- **Workspace**: `TrainDiffusionUnetHybridWorkspaceNoEnv` — epoch-based train loop, EMA, DDPM/DDIM val metrics
- **Policy**: `DiffusionAttentionHybridImagePolicy` — ResNet18 image encoder + cross-attention conditioned UNet1D
- **Dataset**: `PlanarPushingAttentionDataset` — multi-zarr, variable-length obs windows, ImprovedDatasetSampler
- **Noise model**: `AttentionConditionalUnet1D` — cross-attends to observation tokens with temporal position encoding
- **Data format**: Zarr replay buffers at `data/iros/long_context_planar_pushing/*.zarr`

### Observations & Actions
- Obs: `overhead_camera` (3x128x128 RGB), `wrist_camera` (3x128x128 RGB), `agent_pos` (2D)
- Actions: 2D, horizon=24, images cropped to 112x112

### Config Pattern
Configs at `config/iros/long_context_planar_pushing/data_experiments/unet_cross_attention/` specify:
- Multiple zarr datasets with `max_train_episodes` and `val_ratio`
- `training_mode: "random"` for variable obs length sampling
- Checkpoint top-K by `val_loss` and `val_ddim_mse`

## Dependencies
Key: torch, diffusers==0.11.1, hydra-core==1.2.0, robomimic==0.3.0, zarr, wandb, einops
