from typing import Union, Optional
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
    Positional encoding for temporal sequences of observations.
    Handles variable-length sequences where tokens represent observations from different time steps.
    """
    
    def __init__(self, embed_dim: int, max_position: int = 100, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_position = max_position
        self.dropout = nn.Dropout(dropout)
        
        # Create sinusoidal positional encodings
        position = torch.arange(max_position).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_position, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not trainable parameters)
        # NOTE: No scaling here - scaling should be applied to INPUT embeddings, not PE
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None, 
                token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input embeddings, with optional selective application.
        
        Args:
            x: (B, seq_len, embed_dim) - input embeddings
            positions: (B, seq_len) - explicit positions for each token
                      If None, assumes positions are [0, 1, 2, ..., seq_len-1]
            token_mask: (B, seq_len) - True for tokens that should get PE, False otherwise
                       If None, all tokens get PE
        
        Returns:
            x_with_pos: (B, seq_len, embed_dim) - embeddings with positional encoding
        """
        B, seq_len, embed_dim = x.shape
        
        if positions is not None:
            # Use explicit positions provided
            # Clamp positions to valid range to prevent indexing errors
            positions = torch.clamp(positions, 0, self.max_position - 1)
            pos_encodings = self.pe[positions.long()]  # (B, seq_len, embed_dim)
        else:
            # Default sequential positions [0, 1, 2, ..., seq_len-1]
            pos_encodings = self.pe[:seq_len].unsqueeze(0).expand(B, -1, -1)  # (B, seq_len, embed_dim)
        
        # Apply positional encoding selectively if mask is provided
        if token_mask is not None:
            # Only add PE where mask is True - more efficient than clone + subtract
            pos_encodings = pos_encodings * token_mask.unsqueeze(-1).float()
        x_with_pos = x + pos_encodings
        return self.dropout(x_with_pos)

class VariableLengthGlobalAttention(nn.Module):
    """Cross-attention module that allows trajectory timesteps to attend to variable-length global context."""
    
    def __init__(self, 
                 embed_dim: int,
                 global_cond_dim: int, 
                 max_global_tokens: int = 5,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_layer_norm: bool = True,
                 use_temporal_pos_encoding: bool = True,
                 max_temporal_position: int = 1000):  # Increased to accommodate larger position ranges
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_global_tokens = max_global_tokens
        self.use_temporal_pos_encoding = use_temporal_pos_encoding
        
        # Project global conditioning to this layer's attention space
        self.global_proj = nn.Linear(global_cond_dim, embed_dim)
        
        # Temporal positional encoding ONLY for observation tokens
        if use_temporal_pos_encoding:
            self.temporal_pos_encoding = TemporalPositionalEncoding(
                embed_dim=embed_dim,
                max_position=max_temporal_position,
                dropout=dropout
            )
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization (pre-norm style)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.query_norm = nn.LayerNorm(embed_dim)
            self.key_value_norm = nn.LayerNorm(embed_dim)
        
        # Feedforward network after attention (standard transformer practice)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
        if use_layer_norm:
            self.ffn_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, global_cond: torch.Tensor, 
                global_mask: Optional[torch.Tensor] = None,
                temporal_positions: Optional[torch.Tensor] = None,
                token_type_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, embed_dim) - trajectory timestep features
            global_cond: (B, max_tokens, global_cond_dim) - global conditioning tokens
            global_mask: (B, max_tokens) - True for valid tokens, False for padding
            temporal_positions: (B, max_tokens) - temporal positions for observation tokens only
            token_type_mask: (B, max_tokens) - True for observation tokens that need temporal PE, False for special tokens
        
        Returns:
            output: (B, T, embed_dim) - attended trajectory features
            attention_weights: (B, num_heads, T, max_tokens) - attention weights for interpretability
        """
        B, T, embed_dim = x.shape
        
        # Project global conditioning to attention space
        global_features = self.global_proj(global_cond)  # (B, max_tokens, embed_dim)
        
        # Scale projected embeddings (standard transformer practice)
        global_features = global_features * math.sqrt(self.embed_dim)
        
        # Apply temporal positional encoding ONLY to observation tokens (efficient)
        if self.use_temporal_pos_encoding and token_type_mask is not None:
            # Use the improved positional encoding that supports selective application
            obs_token_mask = token_type_mask  # True for observation tokens
            
            if torch.any(obs_token_mask):
                # Apply PE directly with masking - no cloning or subtraction needed
                global_features = self.temporal_pos_encoding(
                    global_features,        # (B, max_tokens, embed_dim)
                    temporal_positions,     # (B, max_tokens)
                    token_mask=obs_token_mask  # (B, max_tokens) - only obs tokens get PE
                )
        
        # Apply layer normalization (pre-norm)
        if self.use_layer_norm:
            x_normed = self.query_norm(x)
            global_features_normed = self.key_value_norm(global_features)
        else:
            x_normed = x
            global_features_normed = global_features
        
        # Cross-attention: trajectory attends to global context
        attended_x, attention_weights = self.cross_attention(
            query=x_normed,                                    # (B, T, embed_dim)
            key=global_features_normed,                        # (B, max_tokens, embed_dim)
            value=global_features_normed,                      # (B, max_tokens, embed_dim)
            key_padding_mask=~global_mask if global_mask is not None else None,  # Mask padded tokens
            average_attn_weights=False  # Return per-head attention weights
        )
        
        # First residual connection
        x = x + attended_x
        
        # Feedforward network with second residual connection
        if self.use_layer_norm:
            ffn_input = self.ffn_norm(x)
        else:
            ffn_input = x
        
        ffn_output = self.ffn(ffn_input)
        x = x + ffn_output
        
        return x, attention_weights

class AttentionConditionalResidualBlock1D(nn.Module):
    """Residual block that uses cross-attention for variable-length token conditioning."""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 global_cond_dim: int,
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 max_global_tokens: int = 16,  # Total token capacity (including special tokens)
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 max_obs_position: Optional[int] = None):  # Observation window size
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Convolutional blocks (same as original)
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        
        # Residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        # Set default observation window size if not provided
        if max_obs_position is None:
            # For backwards compatibility, subtract special tokens from total capacity
            max_obs_position = max_global_tokens - 2 if max_global_tokens > 2 else 1
        
        # Cross-attention conditioning with CORRECT design: max_temporal_position = observation window
        self.attention_conditioning = VariableLengthGlobalAttention(
            embed_dim=out_channels,
            global_cond_dim=global_cond_dim,
            max_global_tokens=max_global_tokens,  # Total token capacity (includes special tokens)
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            use_layer_norm=True,
            use_temporal_pos_encoding=True,
            max_temporal_position=max_obs_position  # Observation window size only
        )
        
    def forward(self, x: torch.Tensor, global_cond: torch.Tensor, 
                global_mask: Optional[torch.Tensor] = None,
                temporal_positions: Optional[torch.Tensor] = None,
                token_type_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, in_channels, T) - input trajectory features  
            global_cond: (B, max_tokens, global_cond_dim) - global conditioning tokens
            global_mask: (B, max_tokens) - mask for valid conditioning tokens
            temporal_positions: (B, max_tokens) - temporal positions of observations
            token_type_mask: (B, max_tokens) - True for observation tokens, False for special tokens
        """
        # Apply first conv block
        out = self.blocks[0](x)
        
        # Apply cross-attention conditioning between conv blocks
        # Reshape for attention: (B, out_channels, T) -> (B, T, out_channels)
        out_for_attention = out.transpose(1, 2)  # (B, T, out_channels)
        # Apply attention conditioning with selective positional encoding
        attended_features, attention_weights = self.attention_conditioning(
            out_for_attention, global_cond, global_mask, temporal_positions, token_type_mask
        )
        # Reshape back: (B, T, out_channels) -> (B, out_channels, T)
        attended_features = attended_features.transpose(1, 2)
        
        # Combine attention conditioning with conv features
        out = out + attended_features
        # Second conv block
        out = self.blocks[1](out)
        # Residual connection
        out = out + self.residual_conv(x)
        
        return out

class AttentionConditionalUnet1D(nn.Module):
    """1D U-Net with attention-based conditioning for variable-length token sequences."""
    
    def __init__(self,
                 input_dim: int,
                 global_cond_dim: Optional[int] = None,
                 target_dim: Optional[int] = None,
                 local_cond_dim: Optional[int] = None,
                 diffusion_step_embed_dim: int = 256,
                 down_dims: list = [256, 512, 1024],
                 kernel_size: int = 3,
                 n_groups: int = 8,
                 max_global_tokens: int = 16,
                 num_attention_heads: int = 8,
                 attention_dropout: float = 0.1,
                 use_target_conditioning: bool = True):
        super().__init__()
        
        self.global_cond_dim = global_cond_dim
        self.target_dim = target_dim
        self.max_global_tokens = max_global_tokens
        self.use_target_conditioning = use_target_conditioning
        
        # CORRECT DESIGN: max_temporal_position = max_global_tokens
        # This creates an observation window where position indices go from 0 to max_global_tokens-1
        # Position 0 = oldest observation in window, Position max_global_tokens-1 = most recent
        max_temporal_position = max_global_tokens
        
        # Calculate total token capacity (observations + special tokens)
        special_token_count = 1  # Always have timestep token
        if use_target_conditioning and target_dim is not None:
            special_token_count += 1  # Add target token if enabled
        max_total_tokens = max_global_tokens + special_token_count
        
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        # Diffusion timestep embedding
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # For the global conditioning, we'll create tokens that combine:
        # - Observation tokens (variable length)
        # - Timestep token (always present) 
        # - Target token (if provided)
        
        # Project all conditioning to a unified dimension for attention
        unified_token_dim = down_dims[0]  # Use first down dim as unified dimension
        
        # Timestep projection to token space
        self.timestep_to_token = nn.Linear(dsed, unified_token_dim)
        
        # Global conditioning projection to token space
        if global_cond_dim is not None:
            self.global_to_token = nn.Linear(global_cond_dim, unified_token_dim)
        else:
            self.global_to_token = None
            
        # Target conditioning projection to token space (only if enabled)
        if use_target_conditioning and target_dim is not None:
            self.target_to_token = nn.Linear(target_dim, unified_token_dim)
        else:
            self.target_to_token = None
        
        # Build network layers
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        # Local conditioning encoder (using attention-based blocks)
        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            local_cond_encoder = nn.ModuleList([
                AttentionConditionalResidualBlock1D(
                    local_cond_dim, dim_out, 
                    global_cond_dim=unified_token_dim,  # Uses unified token dimension
                    kernel_size=kernel_size, n_groups=n_groups,
                    max_global_tokens=max_total_tokens,  # Total token capacity (includes special tokens)
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    max_obs_position=max_temporal_position),  # Observation window size
                AttentionConditionalResidualBlock1D(
                    local_cond_dim, dim_out,
                    global_cond_dim=unified_token_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    max_global_tokens=max_total_tokens,  # Total token capacity (includes special tokens)
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    max_obs_position=max_temporal_position)  # Observation window size
            ])

        # Middle modules
        mid_dim = all_dims[-1]
        mid_modules = nn.ModuleList([
            AttentionConditionalResidualBlock1D(
                mid_dim, mid_dim, 
                global_cond_dim=unified_token_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                max_global_tokens=max_global_tokens,  # FIXED: Use original max_global_tokens
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout
            ),
            AttentionConditionalResidualBlock1D(
                mid_dim, mid_dim,
                global_cond_dim=unified_token_dim, 
                kernel_size=kernel_size, n_groups=n_groups,
                max_global_tokens=max_global_tokens,  # FIXED: Use original max_global_tokens
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout
            ),
        ])

        # Down modules
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                AttentionConditionalResidualBlock1D(
                    dim_in, dim_out,
                    global_cond_dim=unified_token_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    max_global_tokens=max_total_tokens,  # Total token capacity (includes special tokens)
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    max_obs_position=max_temporal_position),  # Observation window size
                AttentionConditionalResidualBlock1D(
                    dim_out, dim_out,
                    global_cond_dim=unified_token_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    max_global_tokens=max_total_tokens,  # Total token capacity (includes special tokens)
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    max_obs_position=max_temporal_position),  # Observation window size
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # Up modules
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                AttentionConditionalResidualBlock1D(
                    dim_out*2, dim_in,
                    global_cond_dim=unified_token_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    max_global_tokens=max_total_tokens,  # Total token capacity (includes special tokens)
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    max_obs_position=max_temporal_position),  # Observation window size
                AttentionConditionalResidualBlock1D(
                    dim_in, dim_in,
                    global_cond_dim=unified_token_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    max_global_tokens=max_total_tokens,  # Total token capacity (includes special tokens)
                    num_attention_heads=num_attention_heads,
                    attention_dropout=attention_dropout,
                    max_obs_position=max_temporal_position),  # Observation window size
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        # Final convolution
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.mid_modules = mid_modules
        self.final_conv = final_conv
        self.unified_token_dim = unified_token_dim

        logger.info(
            "AttentionConditionalUnet1D - number of parameters: %e", 
            sum(p.numel() for p in self.parameters())
        )
    
    def _prepare_tokens(self, 
                        timestep: Union[torch.Tensor, float, int],
                        global_cond: Optional[torch.Tensor] = None,
                        global_mask: Optional[torch.Tensor] = None,
                        target_cond: Optional[torch.Tensor] = None,
                        temporal_positions: Optional[torch.Tensor] = None,
                        batch_size: int = None) -> tuple:
        """
        Prepare conditioning tokens for attention using STANDARD approach:
        - Special tokens (timestep, target) get NO positional encoding
        - Only observation tokens get temporal positional encoding
        
        Returns:
            combined_tokens: (B, max_tokens, unified_token_dim) - All tokens projected to unified dimension
            combined_mask: (B, max_tokens) - Valid token mask 
            combined_temporal_positions: (B, max_tokens) - Temporal positions (only meaningful for obs tokens)
            token_type_mask: (B, max_tokens) - True for observation tokens, False for special tokens
        """
        B = batch_size
        device = global_cond.device if global_cond is not None else (
                target_cond.device if target_cond is not None else torch.device('cpu'))
        
        tokens_list = []
        mask_list = []
        temporal_pos_list = []
        token_type_list = []  # Track which tokens are observations vs special tokens
        
        # 1. Add timestep token (SPECIAL TOKEN - no positional encoding)
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        timesteps = timesteps.expand(B)
        
        timestep_embed = self.diffusion_step_encoder(timesteps)  # (B, dsed)
        timestep_tokens = self.timestep_to_token(timestep_embed).unsqueeze(1)  # (B, 1, unified_dim)
        
        tokens_list.append(timestep_tokens)
        mask_list.append(torch.ones(B, 1, dtype=torch.bool, device=device))
        temporal_pos_list.append(torch.zeros(B, 1, dtype=torch.long, device=device))  # Placeholder (not used)
        token_type_list.append(torch.zeros(B, 1, dtype=torch.bool, device=device))    # False = special token
        
        # 2. Add target token if provided (SPECIAL TOKEN - no positional encoding)
        if target_cond is not None and self.target_to_token is not None:
            target_tokens = self.target_to_token(target_cond).unsqueeze(1)  # (B, 1, unified_dim)
            
            tokens_list.append(target_tokens)
            mask_list.append(torch.ones(B, 1, dtype=torch.bool, device=device))
            temporal_pos_list.append(torch.zeros(B, 1, dtype=torch.long, device=device))  # Placeholder (not used)
            token_type_list.append(torch.zeros(B, 1, dtype=torch.bool, device=device))    # False = special token
        
        # 3. Add observation tokens (TEMPORAL TOKENS - get positional encoding)
        if global_cond is not None and self.global_to_token is not None:
            if len(global_cond.shape) == 2:  # (B, dim) - single observation
                obs_tokens = self.global_to_token(global_cond).unsqueeze(1)  # (B, 1, unified_dim)
                tokens_list.append(obs_tokens)
                mask_list.append(torch.ones(B, 1, dtype=torch.bool, device=device))
                
                # Default temporal position for single observation
                default_pos = torch.ones(B, 1, dtype=torch.long, device=device)
                temporal_pos_list.append(default_pos)
                token_type_list.append(torch.ones(B, 1, dtype=torch.bool, device=device))  # True = observation token
                
            else:  # (B, num_tokens, dim) - multiple observations
                obs_tokens = self.global_to_token(global_cond)  # (B, num_tokens, unified_dim)
                tokens_list.append(obs_tokens)
                
                # Use provided mask or default to all valid
                if global_mask is not None:
                    mask_list.append(global_mask)
                else:
                    num_tokens = global_cond.shape[1]
                    mask_list.append(torch.ones(B, num_tokens, dtype=torch.bool, device=device))
                
                # Use provided temporal positions
                if temporal_positions is not None:
                    temporal_pos_list.append(temporal_positions)
                else:
                    # Default sequential positions starting from 0
                    num_tokens = global_cond.shape[1]
                    default_positions = torch.arange(num_tokens, device=device).unsqueeze(0).expand(B, -1)
                    temporal_pos_list.append(default_positions)
                
                # All observation tokens need positional encoding
                num_tokens = global_cond.shape[1]
                token_type_list.append(torch.ones(B, num_tokens, dtype=torch.bool, device=device))  # True = observation
        
        # 4. Concatenate all tokens
        combined_tokens = torch.cat(tokens_list, dim=1)  # (B, total_tokens, unified_dim)
        combined_mask = torch.cat(mask_list, dim=1)      # (B, total_tokens)
        combined_temporal_positions = torch.cat(temporal_pos_list, dim=1)  # (B, total_tokens)
        token_type_mask = torch.cat(token_type_list, dim=1)  # (B, total_tokens)
        
        # 5. Pad to max_total_tokens if necessary
        current_tokens = combined_tokens.shape[1]
        # Calculate total capacity same way as in constructor
        special_token_count = 1  # Always have timestep token
        if self.use_target_conditioning and self.target_dim is not None:
            special_token_count += 1  # Add target token if enabled
        max_total_tokens = self.max_global_tokens + special_token_count

        assert current_tokens == max_total_tokens
        
        # if current_tokens < max_total_tokens:
        #     pad_tokens = max_total_tokens - current_tokens
            
        #     # Pad with zeros
        #     token_pad = torch.zeros(B, pad_tokens, self.unified_token_dim, device=device)
        #     mask_pad = torch.zeros(B, pad_tokens, dtype=torch.bool, device=device)
        #     pos_pad = torch.zeros(B, pad_tokens, dtype=torch.long, device=device)
        #     type_pad = torch.zeros(B, pad_tokens, dtype=torch.bool, device=device)  # Padding = special token
        #     type_pad[:, 0] = 1  # First token is always the timestep token
        #     if self.use_target_conditioning and self.target_dim is not None:
        #         type_pad[:, 1] = 1  # Second token is the target token

        #     combined_tokens = torch.cat([combined_tokens, token_pad], dim=1)
        #     combined_mask = torch.cat([combined_mask, mask_pad], dim=1)
        #     combined_temporal_positions = torch.cat([combined_temporal_positions, pos_pad], dim=1)
        #     token_type_mask = torch.cat([token_type_mask, type_pad], dim=1)
        
        return combined_tokens, combined_mask, combined_temporal_positions, token_type_mask

    def forward(self,
                sample: torch.Tensor,
                timestep: Union[torch.Tensor, float, int],
                global_cond: Optional[torch.Tensor] = None,
                global_mask: Optional[torch.Tensor] = None,
                target_cond: Optional[torch.Tensor] = None,
                local_cond: Optional[torch.Tensor] = None,
                temporal_positions: Optional[torch.Tensor] = None,
                **kwargs):
        """
        Forward pass using attention-based token conditioning.
        
        Args:
            sample: (B, T, input_dim) - input trajectory
            timestep: (B,) or scalar - diffusion timestep
            global_cond: (B, global_cond_dim) or (B, num_tokens, global_cond_dim) - global conditioning
            global_mask: (B, num_tokens) - mask for valid global tokens
            target_cond: (B, target_dim) - target/goal conditioning
            local_cond: (B, T, local_cond_dim) - per-timestep local conditioning
            temporal_positions: (B, num_tokens) - temporal positions of observations
            
        Returns:
            output: (B, T, input_dim) - denoised trajectory
        """
        # Rearrange input: (B, T, input_dim) -> (B, input_dim, T)
        sample = einops.rearrange(sample, 'b h t -> b t h')
        B, T, input_dim = sample.shape

        # Prepare conditioning tokens with selective positional encoding
        combined_tokens, combined_mask, combined_positions, token_type_mask = self._prepare_tokens(
            timestep, global_cond, global_mask, target_cond, temporal_positions, B
        )
        
        # Handle local conditioning
        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, combined_tokens, combined_mask, combined_positions, token_type_mask)
            h_local.append(x)
            x = resnet2(local_cond, combined_tokens, combined_mask, combined_positions, token_type_mask)
            h_local.append(x)
        
        # Forward pass through U-Net
        x = sample
        h = []
        
        # Downsampling path
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, combined_tokens, combined_mask, combined_positions, token_type_mask)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, combined_tokens, combined_mask, combined_positions, token_type_mask)
            h.append(x)
            x = downsample(x)

        # Middle layers
        for mid_module in self.mid_modules:
            x = mid_module(x, combined_tokens, combined_mask, combined_positions, token_type_mask)

        # Upsampling path
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, combined_tokens, combined_mask, combined_positions, token_type_mask)
            # Note: keeping original bug for compatibility
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, combined_tokens, combined_mask, combined_positions, token_type_mask)
            x = upsample(x)

        # Final convolution
        x = self.final_conv(x)

        # Rearrange output: (B, input_dim, T) -> (B, T, input_dim)
        x = einops.rearrange(x, 'b t h -> b h t')
        
        return x


# ================================================================================================
# USAGE EXAMPLE AND EXPLANATION
# ================================================================================================

"""
IMPROVED USAGE GUIDELINES AND BEST PRACTICES:

## KEY INSIGHT: CONSISTENT TEMPORAL POSITIONING
The most important principle is that observations should have the SAME positional 
encoding regardless of how many total tokens are used in conditioning.

Example: 16-step observation history [obs_0, obs_1, ..., obs_15]

Case 1: Condition on 5 most recent observations
    global_cond: (B, 5, obs_dim) = [obs_11, obs_12, obs_13, obs_14, obs_15]
    temporal_positions: (B, 5) = [11, 12, 13, 14, 15]
    global_mask: (B, 5) = [True, True, True, True, True]

Case 2: Condition on 2 most recent observations  
    global_cond: (B, 5, obs_dim) = [obs_14, obs_15, PAD, PAD, PAD]
    temporal_positions: (B, 5) = [14, 15, 0, 0, 0]  # Same positions 14,15!
    global_mask: (B, 5) = [True, True, False, False, False]

## CRITICAL: obs_14 and obs_15 get IDENTICAL positional encodings in both cases!

## HANDLING MIXED BATCHES
You can have different conditioning lengths within the same batch:

Batch example:
- Sample 0: Use 5 tokens → temporal_positions[0] = [11, 12, 13, 14, 15]
- Sample 1: Use 3 tokens → temporal_positions[1] = [13, 14, 15, 0, 0] 
- Sample 2: Use 2 tokens → temporal_positions[2] = [14, 15, 0, 0, 0]
- Sample 3: Use 4 tokens → temporal_positions[3] = [12, 13, 14, 15, 0]

All samples that include obs_15 will have IDENTICAL positional encoding for that observation.

## UTILITY FUNCTIONS
"""

def create_consistent_temporal_batch(observation_sequences: list, 
                                   num_tokens_per_sample: list,
                                   max_tokens: int = 16,
                                   strategy: str = 'most_recent'):
    """
    Create a batch with consistent temporal positioning across variable-length conditioning.
    
    Args:
        observation_sequences: List of tensors, each (seq_len, obs_dim)
        num_tokens_per_sample: List of ints, number of tokens to use per sample
        max_tokens: Maximum tokens per sample (for padding)
        strategy: 'most_recent', 'uniform_sample', or 'oldest_first'
    
    Returns:
        dict with keys: 'global_cond', 'global_mask', 'temporal_positions'
    """
    batch_size = len(observation_sequences)
    if batch_size == 0:
        raise ValueError("Empty observation sequences")
        
    obs_dim = observation_sequences[0].shape[-1]
    
    # Initialize batch tensors
    global_cond = torch.zeros(batch_size, max_tokens, obs_dim)
    global_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    temporal_positions = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    
    for i, (obs_seq, num_tokens) in enumerate(zip(observation_sequences, num_tokens_per_sample)):
        seq_len = obs_seq.shape[0]
        num_tokens = min(num_tokens, seq_len, max_tokens)
        
        if strategy == 'most_recent':
            # Take the most recent observations - RECOMMENDED for consistency
            start_idx = max(0, seq_len - num_tokens)
            selected_obs = obs_seq[start_idx:start_idx + num_tokens]
            # CRITICAL: Use actual temporal positions from original sequence
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
        
        # Fill batch tensors
        global_cond[i, :num_tokens] = selected_obs
        global_mask[i, :num_tokens] = True
        temporal_positions[i, :num_tokens] = positions
    
    return {
        'global_cond': global_cond,
        'global_mask': global_mask, 
        'temporal_positions': temporal_positions
    }

def example_consistent_usage():
    """Demonstrate consistent positioning across different conditioning lengths."""
    
    # Simulate 16-step observation history
    full_obs_sequence = torch.randn(16, 128)  # (16, obs_dim)
    
    # Case 1: Condition on 5 most recent observations
    batch_5 = create_consistent_temporal_batch(
        observation_sequences=[full_obs_sequence],
        num_tokens_per_sample=[5],
        strategy='most_recent'
    )
    print("5-token conditioning:")
    print(f"  Temporal positions: {batch_5['temporal_positions'][0]}")
    print(f"  Global mask: {batch_5['global_mask'][0]}")
    
    # Case 2: Condition on 2 most recent observations
    batch_2 = create_consistent_temporal_batch(
        observation_sequences=[full_obs_sequence], 
        num_tokens_per_sample=[2],
        strategy='most_recent'
    )
    print("\\n2-token conditioning:")
    print(f"  Temporal positions: {batch_2['temporal_positions'][0]}")
    print(f"  Global mask: {batch_2['global_mask'][0]}")
    
    print("\\n✓ IMPORTANT: Positions 14 and 15 are identical in both cases!")
    print("  This ensures consistent positional encoding for the same observations.")
    
    return batch_5, batch_2

def example_mixed_batch():
    """Demonstrate mixed conditioning lengths in a single batch."""
    
    # Different observation sequences
    obs_sequences = [
        torch.randn(16, 128),  # Full 16-step history
        torch.randn(12, 128),  # 12-step history
        torch.randn(8, 128),   # 8-step history
        torch.randn(16, 128)   # Another 16-step history
    ]
    
    # Different conditioning lengths per sample
    num_tokens_list = [5, 3, 2, 4]
    
    mixed_batch = create_consistent_temporal_batch(
        observation_sequences=obs_sequences,
        num_tokens_per_sample=num_tokens_list,
        max_tokens=5,
        strategy='most_recent'
    )
    
    print("Mixed batch temporal positions:")
    for i in range(4):
        valid_positions = mixed_batch['temporal_positions'][i][mixed_batch['global_mask'][i]]
        print(f"  Sample {i}: {valid_positions.tolist()} ({len(valid_positions)} tokens)")
    
    return mixed_batch

# TRAINING BEST PRACTICES:
"""
1. ALWAYS use 'most_recent' strategy for temporal consistency
2. Use absolute temporal positions from the original sequence
3. Mix different conditioning lengths in training batches
4. Start with curriculum learning (short sequences → long sequences)  
5. Ensure proper masking for padded tokens
6. Scale INPUT embeddings (not positional encodings) by sqrt(embed_dim)
7. Use pre-norm layer normalization for better training stability

Example training loop:
```python
model = AttentionConditionalUnet1D(...)

for batch_data in dataloader:
    # Create consistent batch
    processed_batch = create_consistent_temporal_batch(
        observation_sequences=batch_data['obs_sequences'],
        num_tokens_per_sample=batch_data['num_tokens'],  # Can vary per sample
        strategy='most_recent'
    )
    
    # Forward pass
    output = model(
        sample=batch_data['trajectories'],
        timestep=batch_data['timesteps'],
        global_cond=processed_batch['global_cond'],
        global_mask=processed_batch['global_mask'], 
        temporal_positions=processed_batch['temporal_positions']
    )
```
"""

# ================================================================================================
# COMPREHENSIVE TESTS
# ================================================================================================

def run_comprehensive_tests():
    """
    Comprehensive test suite for AttentionConditionalUnet1D.
    Run with: python attention_conditional_unet1d.py
    """
    print("🧪 Running comprehensive tests for AttentionConditionalUnet1D...")
    
    def test_basic_functionality():
        """Test basic forward pass with different conditioning scenarios."""
        print("✓ Testing basic functionality...")
        
        # Model parameters
        input_dim = 7
        global_cond_dim = 128  
        target_dim = 64
        batch_size = 4
        seq_len = 16
        
        # Create model
        model = AttentionConditionalUnet1D(
            input_dim=input_dim,
            global_cond_dim=global_cond_dim,
            target_dim=target_dim,
            down_dims=[256, 512, 1024]
        )
        model.eval()
        
        # Test data
        sample = torch.randn(batch_size, seq_len, input_dim)
        timestep = torch.randint(0, 1000, (batch_size,))
        
        # Test 1: Only timestep conditioning
        output1 = model(sample, timestep)
        assert output1.shape == sample.shape, f"Expected {sample.shape}, got {output1.shape}"
        
        # Test 2: With flat global conditioning
        global_cond = torch.randn(batch_size, global_cond_dim)
        output2 = model(sample, timestep, global_cond=global_cond)
        assert output2.shape == sample.shape
        
        # Test 3: With variable-length global conditioning (3 tokens)
        global_cond_tokens = torch.randn(batch_size, 3, global_cond_dim)
        global_mask = torch.ones(batch_size, 3, dtype=torch.bool)
        temporal_positions = torch.tensor([[13, 14, 15]] * batch_size)
        
        output3 = model(sample, timestep, 
                       global_cond=global_cond_tokens,
                       global_mask=global_mask, 
                       temporal_positions=temporal_positions)
        assert output3.shape == sample.shape
        
        # Test 4: With target conditioning  
        target_cond = torch.randn(batch_size, target_dim)
        output4 = model(sample, timestep, 
                       global_cond=global_cond_tokens,
                       global_mask=global_mask,
                       target_cond=target_cond,
                       temporal_positions=temporal_positions)
        assert output4.shape == sample.shape
        
        print("✅ Basic functionality tests passed!")
        return model

    def test_variable_token_lengths():
        """Test handling of variable-length conditioning within batches."""
        print("✓ Testing variable token lengths...")
        
        model = AttentionConditionalUnet1D(
            input_dim=7, global_cond_dim=128
        )
        model.eval()
        
        batch_size = 3
        sample = torch.randn(batch_size, 16, 7)
        timestep = torch.randint(0, 1000, (batch_size,))
        
        # Test with different token lengths
        # 3 tokens
        global_cond_3 = torch.randn(batch_size, 3, 128)
        mask_3 = torch.ones(batch_size, 3, dtype=torch.bool)
        pos_3 = torch.tensor([[13, 14, 15]] * batch_size)
        output1 = model(sample, timestep, global_cond=global_cond_3, 
                       global_mask=mask_3, temporal_positions=pos_3)
        assert output1.shape == sample.shape
        
        # 5 tokens
        global_cond_5 = torch.randn(batch_size, 5, 128) 
        mask_5 = torch.ones(batch_size, 5, dtype=torch.bool)
        pos_5 = torch.tensor([[11, 12, 13, 14, 15]] * batch_size)
        output2 = model(sample, timestep, global_cond=global_cond_5,
                       global_mask=mask_5, temporal_positions=pos_5)
        assert output2.shape == sample.shape
        
        # Single token
        global_cond_1 = torch.randn(batch_size, 1, 128)
        mask_1 = torch.ones(batch_size, 1, dtype=torch.bool)
        pos_1 = torch.tensor([[15]] * batch_size)
        output3 = model(sample, timestep, global_cond=global_cond_1,
                       global_mask=mask_1, temporal_positions=pos_1)
        assert output3.shape == sample.shape
        
        print("✅ Variable token length tests passed!")

    def test_edge_cases():
        """Test edge cases and error conditions."""
        print("✓ Testing edge cases...")
        
        model = AttentionConditionalUnet1D(input_dim=7, global_cond_dim=128)
        model.eval()
        
        # Test scalar timestep
        sample = torch.randn(2, 8, 7)
        output1 = model(sample, 500)  # Scalar timestep
        assert output1.shape == sample.shape
        
        # Test tensor scalar timestep  
        output2 = model(sample, torch.tensor(500))
        assert output2.shape == sample.shape
        
        # Test with small sequence (but compatible with the architecture)
        small_sample = torch.randn(1, 4, 7)  # Use 4 instead of 2 for down/up sampling compatibility
        output3 = model(small_sample, 100)
        assert output3.shape == small_sample.shape
        
        # Test with variable-length conditioning
        global_cond = torch.randn(1, 2, 128)
        output4 = model(small_sample, 100, global_cond=global_cond)
        assert output4.shape == small_sample.shape
        
        print("✅ Edge case tests passed!")

    def test_device_compatibility():
        """Test CUDA compatibility if available."""
        print("✓ Testing device compatibility...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Testing on device: {device}")
        
        model = AttentionConditionalUnet1D(input_dim=4, global_cond_dim=64)
        model = model.to(device)
        model.eval()
        
        sample = torch.randn(2, 8, 4, device=device)
        timestep = torch.randint(0, 1000, (2,), device=device)
        global_cond = torch.randn(2, 2, 64, device=device)
        global_mask = torch.ones(2, 2, dtype=torch.bool, device=device)
        temporal_positions = torch.tensor([[14, 15], [14, 15]], device=device)
        
        output = model(sample, timestep, global_cond=global_cond,
                      global_mask=global_mask, temporal_positions=temporal_positions)
        assert output.device.type == device.type  # Compare device types instead of exact device objects
        assert output.shape == sample.shape
        
        print("✅ Device compatibility tests passed!")

    def test_gradient_flow():
        """Test that gradients flow properly through the model."""
        print("✓ Testing gradient flow...")
        
        model = AttentionConditionalUnet1D(input_dim=4, global_cond_dim=32, target_dim=16)
        model.train()
        
        sample = torch.randn(2, 8, 4, requires_grad=True)
        timestep = torch.randint(0, 1000, (2,))
        global_cond = torch.randn(2, 2, 32, requires_grad=True)
        global_mask = torch.ones(2, 2, dtype=torch.bool)
        temporal_positions = torch.tensor([[14, 15], [14, 15]])
        target_cond = torch.randn(2, 16, requires_grad=True)
        
        output = model(sample, timestep, 
                      global_cond=global_cond, 
                      global_mask=global_mask,
                      temporal_positions=temporal_positions,
                      target_cond=target_cond)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients exist and are not zero
        assert sample.grad is not None and sample.grad.abs().sum() > 0
        assert global_cond.grad is not None and global_cond.grad.abs().sum() > 0
        assert target_cond.grad is not None and target_cond.grad.abs().sum() > 0
        
        # Check model parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
        
        print("✅ Gradient flow tests passed!")

    def test_attention_patterns():
        """Test that output patterns make sense."""
        print("✓ Testing output patterns...")
        
        model = AttentionConditionalUnet1D(
            input_dim=4, global_cond_dim=32
        )
        model.eval()
        
        # Create structured data 
        sample = torch.randn(1, 8, 4)
        timestep = torch.tensor([500])
        
        # 3 observation tokens 
        global_cond = torch.randn(1, 3, 32)
        global_mask = torch.ones(1, 3, dtype=torch.bool)
        temporal_positions = torch.tensor([[13, 14, 15]])
        
        with torch.no_grad():
            output = model(sample, timestep, global_cond=global_cond,
                          global_mask=global_mask, temporal_positions=temporal_positions)
        
        # Check that output has reasonable magnitude
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        assert output.abs().max() < 100, "Output magnitude too large"
        
        print("✅ Output pattern tests passed!")

    def test_utility_functions():
        """Test utility functions for batch creation."""
        print("✓ Testing utility functions...")
        
        # Test create_consistent_temporal_batch
        obs_sequences = [
            torch.randn(16, 64),  # Full sequence  
            torch.randn(12, 64),  # Shorter sequence
            torch.randn(8, 64)    # Even shorter
        ]
        num_tokens_list = [5, 3, 2]
        
        batch = create_consistent_temporal_batch(
            obs_sequences, num_tokens_list, max_tokens=5, strategy='most_recent'
        )
        
        assert batch['global_cond'].shape == (3, 5, 64)
        assert batch['global_mask'].shape == (3, 5)
        assert batch['temporal_positions'].shape == (3, 5)
        
        # Check temporal consistency
        assert batch['temporal_positions'][0, 4].item() == 15  # Most recent from 16-seq
        assert batch['temporal_positions'][1, 2].item() == 11  # Most recent from 12-seq  
        assert batch['temporal_positions'][2, 1].item() == 7   # Most recent from 8-seq
        
        # Test that the batch can be used with the model
        model = AttentionConditionalUnet1D(input_dim=4, global_cond_dim=64)
        model.eval()
        
        sample = torch.randn(3, 8, 4)
        timestep = torch.randint(0, 1000, (3,))
        
        # Test with variable-length conditioning from utility function
        output = model(sample, timestep, 
                      global_cond=batch['global_cond'],
                      global_mask=batch['global_mask'],
                      temporal_positions=batch['temporal_positions'])
        assert output.shape == sample.shape
        
        print("✅ Utility function tests passed!")

    # Run all tests
    try:
        test_basic_functionality()
        test_variable_token_lengths() 
        test_edge_cases()
        test_device_compatibility()
        test_gradient_flow()
        test_attention_patterns()
        test_utility_functions()
        
        print("\n🎉 ALL TESTS PASSED! The model is working correctly.")
        print("\nModel summary:")
        print("- ✅ Token-based conditioning architecture")
        print("- ✅ Variable-length observation sequences") 
        print("- ✅ Consistent temporal positioning")
        print("- ✅ Proper gradient flow")
        print("- ✅ Device compatibility")
        print("- ✅ Edge case handling")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_comprehensive_tests()


    