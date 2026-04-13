"""
Masked Autoencoder (MAE) for ViT encoder pre-training.

Fine-tunes a DINO-pretrained ViT on robot observation images by masking
75% of patches, encoding the visible 25%, and reconstructing the masked
patches with a lightweight decoder.

After pre-training, export the encoder weights via get_encoder_state_dict()
and load them into the downstream policy via vit_pretrained="path/to/mae_encoder.pth".

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from diffusion_policy.model.vision.vit_obs_encoder import _load_vit_weights_from_file


class MAEModel(nn.Module):
    def __init__(
        self,
        vit_model_name: str = "vit_small_patch16_224",
        pretrained=True,
        img_size: int = 112,
        patch_size: int = 16,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.img_size = img_size

        # --- Encoder: standard timm ViT ---
        pretrained_is_path = isinstance(pretrained, str)
        timm_pretrained = pretrained if isinstance(pretrained, bool) else False

        # Parse patch size from model name if needed
        if f'patch{patch_size}' not in vit_model_name:
            print(f"Warning: patch_size={patch_size} may not match model name '{vit_model_name}'")

        self.encoder = timm.create_model(
            vit_model_name,
            pretrained=timm_pretrained,
            img_size=img_size,
            num_classes=0,
        )

        if pretrained_is_path:
            print(f"Loading ViT weights from: {pretrained}")
            _load_vit_weights_from_file(
                self.encoder, pretrained, img_size, patch_size)

        encoder_embed_dim = self.encoder.embed_dim
        num_patches = (img_size // patch_size) ** 2  # e.g. 49 for 112/16
        self.num_patches = num_patches

        # --- Decoder ---
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder positional embedding (fixed sin-cos, for all patches + CLS)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self._init_decoder_pos_embed(decoder_embed_dim, num_patches)

        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * 3, bias=True)

        print(f"MAE: encoder={vit_model_name} (dim={encoder_embed_dim}), "
              f"decoder depth={decoder_depth} dim={decoder_embed_dim}, "
              f"patches={num_patches}, mask_ratio={mask_ratio}")

    def _init_decoder_pos_embed(self, embed_dim, num_patches):
        """Initialize decoder positional embeddings with 2D sin-cos."""
        grid_size = int(math.sqrt(num_patches))
        assert grid_size * grid_size == num_patches

        pos_embed = _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    def patchify(self, imgs):
        """
        Convert images to patch pixel values.
        imgs: (B, 3, H, W)
        Returns: (B, num_patches, patch_size**2 * 3)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.img_size
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x):
        """
        Reconstruct images from patch pixel values.
        x: (B, num_patches, patch_size**2 * 3)
        Returns: (B, 3, H, W)
        """
        p = self.patch_size
        h = w = int(math.sqrt(x.shape[1]))
        B = x.shape[0]
        x = x.reshape(B, h, w, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, 3, h, p, w, p)
        x = x.reshape(B, 3, h * p, w * p)
        return x

    def random_masking(self, x, mask_ratio):
        """
        Per-sample random masking by per-sample shuffling.
        x: (B, N, D) — patch embeddings (no CLS token)
        Returns:
            x_masked: (B, N_visible, D)
            mask: (B, N) — 0 = keep, 1 = masked
            ids_restore: (B, N) — indices to unshuffle
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, imgs):
        """
        Encode images with masking.
        imgs: (B, 3, H, W)
        Returns:
            latent: (B, N_visible + 1, encoder_dim) — includes CLS
            mask: (B, num_patches) — 0 = keep, 1 = masked
            ids_restore: (B, num_patches)
        """
        # Patch embed
        x = self.encoder.patch_embed(imgs)  # (B, N, D)

        # Add pos embed (spatial only, no CLS yet)
        x = x + self.encoder.pos_embed[:, 1:, :]

        # Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Prepend CLS token with its pos embed
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N_visible+1, D)

        # Apply ViT blocks
        x = getattr(self.encoder, 'pos_drop', nn.Identity())(x)
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """
        Decode: reconstruct masked patches.
        x: (B, N_visible + 1, encoder_dim)
        ids_restore: (B, num_patches)
        Returns: pred (B, num_patches, patch_size**2 * 3)
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, N_visible+1, decoder_dim)

        # Separate CLS and patch tokens
        cls_dec = x[:, :1, :]
        patch_dec = x[:, 1:, :]

        # Insert mask tokens at masked positions
        B = x.shape[0]
        mask_tokens = self.mask_token.expand(
            B, self.num_patches - patch_dec.shape[1], -1)
        x_ = torch.cat([patch_dec, mask_tokens], dim=1)  # (B, N, decoder_dim)
        # Unshuffle to original patch order
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[2]))
        # Prepend CLS
        x = torch.cat([cls_dec, x_], dim=1)  # (B, N+1, decoder_dim)

        # Add decoder positional embedding
        x = x + self.decoder_pos_embed

        # Apply decoder transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predict pixel values (exclude CLS)
        x = self.decoder_pred(x[:, 1:, :])  # (B, N, patch_size**2 * 3)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Compute MSE loss on masked patches only.
        imgs: (B, 3, H, W) — original images
        pred: (B, num_patches, patch_size**2 * 3)
        mask: (B, num_patches) — 1 = masked (compute loss), 0 = visible
        """
        target = self.patchify(imgs)  # (B, N, patch_size**2 * 3)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) — per-patch MSE

        # Average over masked patches only
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, imgs):
        """
        Full MAE forward pass.
        imgs: (B, 3, H, W)
        Returns: loss (scalar), pred (B, N, patch_size**2*3), mask (B, N)
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def get_encoder_state_dict(self):
        """
        Extract just the ViT encoder state_dict with timm-compatible keys.
        Strips the 'encoder.' prefix so keys match what
        _load_vit_weights_from_file() expects.
        """
        prefix = 'encoder.'
        encoder_sd = {}
        for k, v in self.state_dict().items():
            if k.startswith(prefix):
                clean_key = k[len(prefix):]
                encoder_sd[clean_key] = v
        return encoder_sd


# --- 2D sin-cos positional embedding utilities ---

def _get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sin-cos positional embeddings.
    Returns: (grid_size*grid_size (+1 if cls_token), embed_dim)
    """
    import numpy as np
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # (2, gs, gs)
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, gs*gs)

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    embed_dim: output dimension for each position
    grid: (2, N) positions
    Returns: (N, embed_dim)
    """
    import numpy as np
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # (N, D/2)
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # (N, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)  # (N, D)


def _get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension
    pos: (N,) positions
    Returns: (N, embed_dim)
    """
    import numpy as np
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)  # (N, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (N, D)
