"""
Temporal self-attention modules for processing sequences of per-timestep features.
Shared by both ViT-based and ResNet-based policies.
"""

import torch
import torch.nn as nn


class TemporalTransformerBlock(nn.Module):
    """Pre-norm transformer block for temporal attention."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm1(x)
        x = x + self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=key_padding_mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TemporalEncoder(nn.Module):
    """
    Temporal self-attention across per-timestep feature tokens.
    Operates on concatenated features (all cameras + low_dim).
    """

    def __init__(self, dim, depth, num_heads, max_frames=100,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_frames, dim))
        self.blocks = nn.ModuleList([
            TemporalTransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, T, D) per-timestep features
            key_padding_mask: (B, T) True for PADDED positions
        Returns: (B, T, D)
        """
        T = x.shape[1]
        x = x + self.temporal_pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.norm(x)
