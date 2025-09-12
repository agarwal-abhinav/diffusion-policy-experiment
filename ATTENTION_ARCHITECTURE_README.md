# Attention-Based Diffusion Policy Architecture

This document describes the new attention-based diffusion policy architecture that supports variable-length observation sequences with temporal positional encoding.

## Overview

The attention-based architecture addresses key limitations of standard diffusion policies by:

1. **Variable-length observations**: Use between `min_obs_steps` and `n_obs_steps` observations instead of fixed-length sequences
2. **Attention-based conditioning**: Replace concatenation with cross-attention between trajectory and observation tokens
3. **Temporal positional encoding**: Maintain consistent temporal relationships regardless of observation count
4. **Curriculum learning**: Gradually introduce variability in observation lengths during training

## Key Components

### 1. AttentionConditionalUnet1D
- Located: `diffusion_policy/model/diffusion/attention_conditional_unet1d.py`
- Replaces standard UNet with attention-based conditioning
- Supports variable-length token sequences with temporal positional encoding
- Uses cross-attention layers instead of FiLM conditioning

### 2. DiffusionAttentionHybridImagePolicy
- Located: `diffusion_policy/policy/diffusion_attention_hybrid_image_policy.py`
- Attention-based policy that handles variable observation lengths
- Implements curriculum learning for observation count
- Maintains temporal consistency across different observation lengths

### 3. PlanarPushingAttentionDataset
- Located: `diffusion_policy/dataset/planar_pushing_attention_dataset.py`
- Dataset that provides full horizon-length sequences
- Supports attention-based training with proper masking
- Enhanced sampler for variable-length conditioning

### 4. TrainDiffusionAttentionHybridWorkspace
- Located: `diffusion_policy/workspace/train_diffusion_attention_hybrid_workspace.py`
- Training workspace with attention-specific logging and curriculum
- Tracks observation length statistics and curriculum progress
- Enhanced validation with multiple observation lengths

## Key Concepts

### Variable-Length Observations

Traditional approach:
```python
# Always use first n_obs_steps observations
obs_features = obs[:, :n_obs_steps, ...]  # Fixed length
```

Attention approach:
```python
# Use between min_obs_steps and n_obs_steps observations
num_obs_tokens = random.randint(min_obs_steps, n_obs_steps)  
obs_features = obs[:, -num_obs_tokens:, ...]  # Variable length (most recent)
```

### Temporal Consistency

**Critical principle**: Observations must have consistent temporal positions regardless of how many total tokens are used.

Example with 16-step history `[obs_0, obs_1, ..., obs_15]`:

```python
# Case 1: Use 5 most recent observations
tokens = [obs_11, obs_12, obs_13, obs_14, obs_15]
positions = [11, 12, 13, 14, 15]  # Absolute positions

# Case 2: Use 2 most recent observations  
tokens = [obs_14, obs_15]
positions = [14, 15]  # SAME positions 14,15!
```

This ensures `obs_14` and `obs_15` receive identical positional encodings in both cases.

### Curriculum Learning

The model starts with maximum context and gradually learns to work with less:

```python
def _get_current_obs_steps(self, training_step):
    if training_step < curriculum_steps:
        # Early: mostly use max_obs_steps
        if random() < (1.0 - progress):
            return max_obs_steps
        else:
            return random_between(min_obs_steps, max_obs_steps)
    else:
        # Later: uniform sampling
        return random_between(min_obs_steps, max_obs_steps)
```

## Usage

### Basic Training

```bash
# Train with attention-based architecture
python train.py --config-name=train_diffusion_attention_hybrid_example.yaml

# Key parameters to tune:
#   horizon: Total sequence length (e.g., 24)
#   n_obs_steps: Maximum observations to use (e.g., 16)
#   min_obs_steps: Minimum observations to use (e.g., 2)
#   attention_curriculum_steps: Steps for curriculum (e.g., 10000)
```

### Configuration

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_attention_hybrid_image_policy.DiffusionAttentionHybridImagePolicy
  
  # Core parameters
  horizon: 24                    # Total sequence length
  n_obs_steps: 16                # Maximum observation steps  
  min_obs_steps: 2               # Minimum observation steps
  
  # Attention parameters
  max_global_tokens: 18          # Max tokens for attention
  num_attention_heads: 8         # Attention heads
  attention_dropout: 0.1         # Attention dropout
  variable_obs_curriculum: true  # Enable curriculum learning

task:
  dataset:
    _target_: diffusion_policy.dataset.planar_pushing_attention_dataset.PlanarPushingAttentionDataset
    horizon: 24                  # Must match policy
    n_obs_steps: 16              # Must match policy
    min_obs_steps: 2             # Must match policy
    attention_curriculum: true   # Enable curriculum
```

### Testing and Validation

Run comprehensive tests:
```bash
python test_attention_architecture.py
```

Test individual components:
```bash
# Test policy
python -c "from test_attention_architecture import TestAttentionPolicy; t = TestAttentionPolicy(); t.test_policy_initialization()"

# Test dataset  
python -c "from test_attention_architecture import TestAttentionDataset; t = TestAttentionDataset(); t.test_dataset_initialization()"
```

## Architecture Benefits

### 1. Improved Robustness
- Model works with varying amounts of historical context
- Better generalization to different observation availability scenarios
- Reduced brittleness to missing observations

### 2. Enhanced Efficiency
- Use fewer observations when full context isn't needed
- Computational savings during inference
- More efficient attention computation

### 3. Better Temporal Understanding
- Explicit temporal positional encoding
- Consistent temporal relationships
- Improved handling of temporal dynamics

### 4. Training Stability
- Curriculum learning improves convergence
- Progressive difficulty increase
- Better sample efficiency

## Implementation Details

### Attention Mechanism

```python
class VariableLengthGlobalAttention(nn.Module):
    def forward(self, x, global_cond, global_mask, temporal_positions):
        # x: (B, T, embed_dim) - trajectory features
        # global_cond: (B, max_tokens, global_cond_dim) - obs tokens  
        # global_mask: (B, max_tokens) - valid token mask
        # temporal_positions: (B, max_tokens) - temporal positions
        
        # Add temporal positional encoding
        global_features = self.global_proj(global_cond)
        global_features = self.temporal_pos_encoding(global_features, temporal_positions)
        
        # Cross-attention: trajectory attends to observations
        attended_x, attention_weights = self.cross_attention(
            query=x,                      # Trajectory timesteps
            key=global_features,          # Observation tokens
            value=global_features,        # Observation tokens
            key_padding_mask=~global_mask # Mask padded tokens
        )
        
        return attended_x, attention_weights
```

### Token Preparation

```python
def _prepare_tokens(self, timestep, global_cond, temporal_positions):
    # 1. Timestep token (always present)
    timestep_tokens = self.timestep_to_token(timestep_embed)
    
    # 2. Observation tokens (variable length)
    obs_tokens = self.global_to_token(global_cond)
    
    # 3. Target token (if provided)  
    target_tokens = self.target_to_token(target_cond)
    
    # Combine with proper temporal positioning
    combined_tokens = torch.cat([timestep_tokens, obs_tokens, target_tokens], dim=1)
    combined_positions = torch.cat([timestep_pos, temporal_positions, target_pos], dim=1)
    
    return combined_tokens, combined_mask, combined_positions
```

### Error Handling

The architecture includes comprehensive error handling:

1. **Parameter validation**: Ensures `min_obs_steps <= n_obs_steps`
2. **Batch size handling**: Works with any batch size including 1
3. **Insufficient observations**: Gracefully handles when requested tokens > available
4. **Device compatibility**: Proper CUDA/CPU handling
5. **Gradient flow**: Verified gradient propagation through attention layers

## Performance Considerations

### Memory Usage
- Attention scales quadratically with sequence length
- Use `max_global_tokens` to limit memory usage
- Consider gradient checkpointing for very long sequences

### Computational Efficiency
- Use smaller `num_attention_heads` for faster training
- Enable mixed precision with `use_amp: true`
- Use fewer `down_dims` layers if memory constrained

### Curriculum Learning
- Start `attention_curriculum_steps` around 10K for good curriculum
- Monitor `attention/curriculum_progress` in logs
- Adjust based on validation performance trends

## Troubleshooting

### Common Issues

1. **NaN losses**: Check learning rate, may need to reduce for attention layers
2. **Memory errors**: Reduce `max_global_tokens` or batch size
3. **Slow convergence**: Increase `attention_curriculum_steps` or disable curriculum initially
4. **Poor performance**: Ensure temporal positions are consistent across variable lengths

### Debugging Tools

```python
# Check attention patterns
policy.eval()
with torch.no_grad():
    result = policy.predict_action(obs_dict, num_obs_tokens=5)
    
# Visualize attention weights (if model supports it)
attention_weights = policy.model.get_attention_weights()

# Check curriculum progress
print(f"Current obs steps: {policy._get_current_obs_steps()}")
print(f"Training step: {policy.training_step}")
```

### Logging and Monitoring

Key metrics to monitor:
- `attention/current_obs_length`: Current observation length used
- `attention/curriculum_progress`: Curriculum learning progress (0-1)
- `attention/epoch_avg_obs_length`: Average obs length per epoch
- `val_ddpm_mse_{dataset}_obs{N}`: Validation MSE for different obs lengths

## Examples

### Inference with Different Observation Lengths

```python
policy.eval()
with torch.no_grad():
    # Use minimal context (fast, less accurate)
    result_min = policy.predict_action(obs_dict, num_obs_tokens=2)
    
    # Use full context (slower, more accurate)  
    result_max = policy.predict_action(obs_dict, num_obs_tokens=16)
    
    # Use adaptive context based on curriculum
    result_adaptive = policy.predict_action(obs_dict)  # Uses current curriculum
```

### Mixed Batch with Variable Lengths

```python
from diffusion_policy.dataset.planar_pushing_attention_dataset import create_attention_batch_with_variable_obs

# Create batch with different observation lengths
batch = create_attention_batch_with_variable_obs(
    observation_sequences=obs_list,
    action_sequences=action_list,
    num_tokens_per_sample=[2, 5, 8, 16],  # Different lengths per sample
    strategy='most_recent'
)

# Use in training/inference
output = policy(batch['obs'], batch['action'], timesteps)
```

## Future Extensions

1. **Hierarchical attention**: Multi-level attention for very long sequences
2. **Adaptive attention**: Automatically determine optimal observation count
3. **Multimodal tokens**: Separate attention for different observation modalities
4. **Memory-efficient attention**: Linear attention for longer sequences

## References

- Original diffusion policy paper: [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)  
- Attention mechanisms: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Positional encoding: [Transformer architectures](https://arxiv.org/abs/1706.03762)