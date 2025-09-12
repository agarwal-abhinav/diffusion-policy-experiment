from typing import Union, Optional, List, Dict, Any
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import math

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class TemporalPositionalEncoding(nn.Module):
    """
    Improved temporal positional encoding with better handling of variable-length sequences.
    Uses relative positioning to ensure consistency across different conditioning lengths.
    """
    
    def __init__(self, embed_dim: int, max_position: int = 100, dropout: float = 0.1, 
                 scale_factor: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_position = max_position
        self.scale_factor = scale_factor
        self.dropout = nn.Dropout(dropout)
        
        # Create sinusoidal positional encodings
        position = torch.arange(max_position).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_position, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Scale positional encodings (common practice)
        pe = pe * scale_factor
        
        # Register as buffer (not trainable parameters)
        self.register_buffer('pe', pe)
        
        # Optional learned positional embeddings for comparison
        self.use_learned_pos = False  # Can be enabled if needed
        if self.use_learned_pos:
            self.learned_pe = nn.Embedding(max_position, embed_dim)
            nn.init.trunc_normal_(self.learned_pe.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: (B, seq_len, embed_dim) - input embeddings
            positions: (B, seq_len) - explicit absolute positions for each token
        
        Returns:
            x_with_pos: (B, seq_len, embed_dim) - embeddings with positional encoding
        """
        B, seq_len, embed_dim = x.shape
        
        if positions is not None:
            # Clamp positions to valid range
            positions = torch.clamp(positions, 0, self.max_position - 1)
            
            if self.use_learned_pos:
                pos_encodings = self.learned_pe(positions.long())
            else:
                pos_encodings = self.pe[positions.long()]  # (B, seq_len, embed_dim)
        else:
            # Default sequential positions [0, 1, 2, ..., seq_len-1]
            if self.use_learned_pos:
                pos_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(B, -1)
                pos_encodings = self.learned_pe(pos_indices)
            else:
                pos_encodings = self.pe[:seq_len].unsqueeze(0).expand(B, -1, -1)
        
        x_with_pos = x + pos_encodings
        return self.dropout(x_with_pos)

class ImprovedVariableLengthGlobalAttention(nn.Module):
    """
    Improved cross-attention module following modern transformer best practices.
    Supports consistent temporal positioning across variable-length inputs.
    """
    
    def __init__(self, 
                 embed_dim: int,
                 global_cond_dim: int, 
                 max_global_tokens: int = 16,  # Increased for longer sequences
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_dropout: float = 0.1,  # Separate attention dropout
                 use_bias: bool = False,  # Common in modern transformers
                 use_flash_attention: bool = False,  # For efficiency
                 temperature: float = 1.0):  # Attention temperature
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_global_tokens = max_global_tokens
        self.temperature = temperature
        self.use_flash_attention = use_flash_attention
        
        # Improved input projection
        self.global_proj = nn.Linear(global_cond_dim, embed_dim, bias=use_bias)
        
        # Initialize projection weights properly
        nn.init.xavier_uniform_(self.global_proj.weight)
        if use_bias:
            nn.init.zeros_(self.global_proj.bias)
        
        # Temporal positional encoding
        self.temporal_pos_encoding = TemporalPositionalEncoding(
            embed_dim=embed_dim,
            max_position=max_global_tokens * 2,  # Allow for flexibility
            dropout=dropout,
            scale_factor=math.sqrt(embed_dim)  # Scale by sqrt(d_model)
        )
        
        # Multi-head cross-attention with proper initialization
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=use_bias,
            batch_first=True
        )
        
        # Pre-norm layer normalization (better than post-norm)
        self.query_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.key_value_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Feedforward network with better practices
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4, bias=use_bias),
            nn.GELU(),  # Better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim, bias=use_bias),
            nn.Dropout(dropout)
        )
        
        # Initialize FFN weights
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if use_bias:
                    nn.init.zeros_(layer.bias)
        
        self.ffn_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Optional: Learnable temperature for attention
        self.learnable_temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, x: torch.Tensor, global_cond: torch.Tensor, 
                global_mask: Optional[torch.Tensor] = None,
                temporal_positions: Optional[torch.Tensor] = None) -> tuple:
        """
        Improved forward pass with better attention handling.
        """
        B, T, embed_dim = x.shape
        
        # Project and normalize global conditioning
        global_features = self.global_proj(global_cond)
        
        # Add temporal positional encoding
        global_features = self.temporal_pos_encoding(global_features, temporal_positions)
        
        # Pre-norm (better than post-norm)
        x_normed = self.query_norm(x)
        global_features_normed = self.key_value_norm(global_features)
        
        # Scale attention by learnable temperature
        attention_scale = 1.0 / (self.learnable_temperature * math.sqrt(self.head_dim))
        
        # Cross-attention with improved masking
        attended_x, attention_weights = self.cross_attention(
            query=x_normed,
            key=global_features_normed,
            value=global_features_normed,
            key_padding_mask=~global_mask if global_mask is not None else None,
            average_attn_weights=False,
            need_weights=True
        )
        
        # First residual connection with proper scaling
        x = x + attended_x
        
        # Feedforward network with pre-norm
        ffn_input = self.ffn_norm(x)
        ffn_output = self.ffn(ffn_input)
        
        # Second residual connection  
        x = x + ffn_output
        
        return x, attention_weights

class VariableLengthBatchHandler:
    """
    Utility class for handling batches with variable-length conditioning.
    Ensures consistent temporal positioning across different sequence lengths.
    """
    
    def __init__(self, max_obs_history: int = 16, max_tokens_per_batch: int = 16):
        self.max_obs_history = max_obs_history
        self.max_tokens_per_batch = max_tokens_per_batch
    
    def create_consistent_batch(self, 
                              observations_list: List[torch.Tensor],
                              num_tokens_list: List[int],
                              strategy: str = 'most_recent') -> Dict[str, torch.Tensor]:
        """
        Create a batch with consistent temporal positioning.
        
        Args:
            observations_list: List of observation sequences, each (seq_len, obs_dim)
            num_tokens_list: List of number of tokens to use from each sequence
            strategy: 'most_recent', 'uniform_sample', 'oldest_first'
        
        Returns:
            Dictionary with 'global_cond', 'global_mask', 'temporal_positions'
        """
        batch_size = len(observations_list)
        obs_dim = observations_list[0].shape[-1]
        
        # Initialize batch tensors
        global_cond = torch.zeros(batch_size, self.max_tokens_per_batch, obs_dim)
        global_mask = torch.zeros(batch_size, self.max_tokens_per_batch, dtype=torch.bool)
        temporal_positions = torch.zeros(batch_size, self.max_tokens_per_batch, dtype=torch.long)
        
        for i, (obs_seq, num_tokens) in enumerate(zip(observations_list, num_tokens_list)):
            seq_len = obs_seq.shape[0]
            num_tokens = min(num_tokens, seq_len, self.max_tokens_per_batch)
            
            if strategy == 'most_recent':
                # Take the most recent observations
                start_idx = max(0, seq_len - num_tokens)
                selected_obs = obs_seq[start_idx:start_idx + num_tokens]
                # Temporal positions reflect actual positions in the sequence
                positions = torch.arange(start_idx, start_idx + num_tokens)
                
            elif strategy == 'uniform_sample':
                # Uniformly sample across the sequence
                indices = torch.linspace(0, seq_len - 1, num_tokens, dtype=torch.long)
                selected_obs = obs_seq[indices]
                positions = indices
                
            elif strategy == 'oldest_first':
                # Take the oldest observations
                selected_obs = obs_seq[:num_tokens]
                positions = torch.arange(num_tokens)
            
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Fill batch tensors
            global_cond[i, :num_tokens] = selected_obs
            global_mask[i, :num_tokens] = True
            temporal_positions[i, :num_tokens] = positions
        
        return {
            'global_cond': global_cond,
            'global_mask': global_mask,
            'temporal_positions': temporal_positions
        }
    
    def create_curriculum_batch(self,
                              observations_list: List[torch.Tensor],
                              curriculum_step: int,
                              max_curriculum_steps: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Create batch with curriculum learning for variable sequence lengths.
        Start with short sequences, gradually increase length.
        """
        # Curriculum: start with 1-2 tokens, gradually increase to max
        min_tokens = 1
        max_tokens = min(self.max_tokens_per_batch, self.max_obs_history)
        
        # Progressive increase in sequence length
        progress = min(curriculum_step / max_curriculum_steps, 1.0)
        current_max_tokens = min_tokens + int(progress * (max_tokens - min_tokens))
        
        # Sample number of tokens for each sequence
        num_tokens_list = []
        for _ in range(len(observations_list)):
            num_tokens = torch.randint(min_tokens, current_max_tokens + 1, (1,)).item()
            num_tokens_list.append(num_tokens)
        
        return self.create_consistent_batch(
            observations_list, num_tokens_list, strategy='most_recent'
        )

# Usage examples and training strategies
class TrainingExample:
    """Examples of how to use the improved attention conditioning."""
    
    @staticmethod
    def example_consistent_positioning():
        """
        Example showing consistent positioning across variable lengths.
        Key insight: 2 most recent tokens should have same positions regardless
        of whether we're conditioning on 2 or 5 tokens.
        """
        batch_handler = VariableLengthBatchHandler(max_obs_history=16, max_tokens_per_batch=5)
        
        # Example: 16-step observation history
        full_obs_sequence = torch.randn(16, 128)  # (16, obs_dim)
        
        # Case 1: Condition on all 5 tokens (most recent)
        batch_5_tokens = batch_handler.create_consistent_batch(
            observations_list=[full_obs_sequence],
            num_tokens_list=[5],
            strategy='most_recent'
        )
        # Results in: temporal_positions = [11, 12, 13, 14, 15] (last 5 positions)
        
        # Case 2: Condition on 2 tokens (most recent)  
        batch_2_tokens = batch_handler.create_consistent_batch(
            observations_list=[full_obs_sequence],
            num_tokens_list=[2], 
            strategy='most_recent'
        )
        # Results in: temporal_positions = [14, 15, 0, 0, 0] (last 2 positions + padding)
        
        # Key insight: Positions 14 and 15 have identical positional encodings
        # in both cases, ensuring consistency!
        
        print("5-token case temporal positions:", batch_5_tokens['temporal_positions'])
        print("2-token case temporal positions:", batch_2_tokens['temporal_positions'])
        return batch_5_tokens, batch_2_tokens
    
    @staticmethod
    def example_mixed_batch_training():
        """Example of training with mixed sequence lengths in the same batch."""
        batch_handler = VariableLengthBatchHandler()
        
        # Different observation sequences
        obs_sequences = [
            torch.randn(16, 128),  # Full history
            torch.randn(12, 128),  # Partial history
            torch.randn(8, 128),   # Short history
            torch.randn(16, 128)   # Full history again
        ]
        
        # Different conditioning lengths
        num_tokens_list = [5, 3, 2, 4]  # Variable tokens per sample
        
        batch_data = batch_handler.create_consistent_batch(
            observations_list=obs_sequences,
            num_tokens_list=num_tokens_list,
            strategy='most_recent'
        )
        
        print("Batch shape:", batch_data['global_cond'].shape)
        print("Temporal positions:\n", batch_data['temporal_positions'])
        print("Global mask:\n", batch_data['global_mask'])
        
        return batch_data

# Best practices for training
"""
TRAINING BEST PRACTICES:

1. CONSISTENT TEMPORAL POSITIONING:
   - Always use absolute temporal positions from the original sequence
   - 2 most recent observations should have same positions (e.g., [14,15]) 
     whether you condition on 2 or 5 tokens
   - Use VariableLengthBatchHandler.create_consistent_batch()

2. CURRICULUM LEARNING:
   - Start training with shorter sequences (1-2 tokens)
   - Gradually increase to longer sequences
   - Use create_curriculum_batch() for progressive training

3. BATCH COMPOSITION:
   - Mix different sequence lengths in the same batch
   - Ensure proper masking for padded positions
   - Use most_recent strategy for temporal consistency

4. ATTENTION TEMPERATURE:
   - Start with temperature=1.0, can be learned or annealed
   - Lower temperature = sharper attention (more focused)
   - Higher temperature = smoother attention (more distributed)

5. POSITIONAL ENCODING SCALING:
   - Scale by sqrt(embed_dim) for better gradient flow
   - Consider learnable positional embeddings for fine-tuning

6. REGULARIZATION:
   - Use separate dropouts for attention and feedforward
   - Pre-norm is more stable than post-norm
   - Gradient clipping recommended for training stability

Example training loop:
```python
model = AttentionConditionalUnet1D(...)
batch_handler = VariableLengthBatchHandler()

for epoch in range(num_epochs):
    for batch_idx, raw_batch in enumerate(dataloader):
        # Create consistent variable-length batch
        processed_batch = batch_handler.create_consistent_batch(
            observations_list=raw_batch['observations'],
            num_tokens_list=raw_batch['num_tokens'],
            strategy='most_recent'
        )
        
        # Forward pass
        output = model(
            sample=raw_batch['trajectories'],
            timestep=raw_batch['timesteps'], 
            global_cond=processed_batch['global_cond'],
            global_mask=processed_batch['global_mask'],
            temporal_positions=processed_batch['temporal_positions']
        )
        
        # Compute loss and backprop...
```
"""