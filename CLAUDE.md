# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

**Conda Environment**: Use `conda_environment.yaml` for Linux or `conda_environment_macos.yaml` for macOS
```bash
# Linux with GPU support
conda env create -f conda_environment.yaml
conda activate robodiff

# macOS (limited functionality)
conda env create -f conda_environment_macos.yaml
```

**Additional Dependencies**: Some packages are installed via pip after conda environment creation (as specified in the environment files).

## Core Training Commands

**Single Seed Training**:
```bash
python train.py --config-dir=. --config-name=<config_name>.yaml \
    training.seed=42 \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

**Multi-Seed Training with Ray**:
```bash
# Start ray cluster first
export CUDA_VISIBLE_DEVICES=0,1,2
ray start --head --num-gpus=3

# Run multi-seed training
python ray_train_multirun.py --config-name=<workspace_config> --seeds=42,43,44 --monitor_key=test/mean_score
```

**Evaluation**:
```bash
python eval.py --checkpoint <path_to_checkpoint.ckpt> --output_dir <output_dir> --device cuda:0
```

## Testing

**Run Tests**: Tests are located in the `tests/` directory. Use Python to run individual tests:
```bash
python -m pytest tests/
# Or run specific tests
python tests/test_replay_buffer.py
```

## Project Architecture

### Core Components

**Workspace Pattern**: The codebase uses a workspace-based architecture where each training method is implemented as a `Workspace` class that inherits from `BaseWorkspace`. Workspaces manage the complete training lifecycle including:
- Model instantiation and training loops
- Checkpoint saving/loading
- Evaluation orchestration
- Logging and metrics tracking

**Task-Method Separation**: The architecture maintains `O(N+M)` complexity for `N` tasks and `M` methods by keeping them independent:

**Task Side**:
- `Dataset`: Adapts datasets to unified interface (returns obs/action dicts)
- `EnvRunner`: Executes policies and produces evaluation metrics  
- `config/task/<task_name>.yaml`: Task configuration files
- (Optional) `Env`: Gym-compatible environment wrapper

**Method Side**:
- `Policy`: Implements inference (`predict_action`) and training (`compute_loss`)
- `Workspace`: Manages training/evaluation lifecycle
- `config/<workspace_name>.yaml`: Method configuration files

### Data Interfaces

**Low-Dim Interface**:
- Policy Input: `{"obs": Tensor(B,To,Do)}`
- Policy Output: `{"action": Tensor(B,Ta,Da)}`
- Dataset Output: `{"obs": Tensor(To,Do), "action": Tensor(Ta,Da)}`

**Image Interface**:
- Policy Input: `{"key0": Tensor(B,To,*), "key1": Tensor(B,To,H,W,3)}`
- Policy Output: `{"action": Tensor(B,Ta,Da)}`
- Dataset Output: `{"obs": {"key0": Tensor(To,*), "key1": Tensor(To,H,W,3)}, "action": Tensor(Ta,Da)}`

Where `To` = observation horizon, `Ta` = action horizon, `T` = prediction horizon.

### Key Data Structures

**ReplayBuffer**: Zarr-based storage for demonstration data with chunking/compression support. Stored as nested directories (`.zarr`) or zip files (`.zarr.zip`).

**SharedMemoryRingBuffer**: Lock-free FILO data structure used in real robot implementations for multi-process communication without pickle overhead.

**LinearNormalizer**: Handles observation/action normalization across the policy interface. Normalization parameters are saved with policy checkpoints.

### Configuration System

The codebase uses Hydra for configuration management:
- `diffusion_policy/config/`: Contains all configuration files
- Task configs: `config/task/<task_name>.yaml`
- Workspace configs: `config/<workspace_name>.yaml`
- Configs are composable via Hydra's override system

### Training Output Structure

Training outputs follow a standardized directory structure:
```
data/outputs/YYYY.MM.DD/HH.MM.SS_<method>_<task>/
├── checkpoints/
│   ├── epoch=XXXX-test_mean_score=X.XXX.ckpt
│   └── latest.ckpt
├── .hydra/
│   └── config.yaml
├── logs.json.txt
└── train.log
```

## Real Robot Support

The codebase includes extensive real robot capabilities:
- Multi-camera RealSense support
- UR5/UR5e robot control via RTDE
- SpaceMouse teleoperation
- Asynchronous policy execution (`RealEnv` splits gym's `step` into `get_obs` and `exec_actions`)

## Development Notes

- **Testing**: Use the existing test files in `tests/` as examples for new functionality
- **Data Paths**: Most paths are configurable via Hydra - use `task.dataset.zarr_path` to specify data locations
- **GPU Memory**: Training uses EMA models and can be memory intensive - monitor GPU usage
- **Wandb Integration**: Training automatically logs to Weights & Biases if configured
- **Evaluation Frequency**: Policies are evaluated every 50 epochs by default during training
- **Multirun Metrics**: Use `multirun_metrics.py` to aggregate results across multiple training runs