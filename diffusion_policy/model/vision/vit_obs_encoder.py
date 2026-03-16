"""
ViT-based observation encoder for diffusion policy.

Supports 8 variants:
  A-small:      Per-frame ViT, CLS projected to small dim
  A-full:       Per-frame ViT, full CLS dim
  B:            ViT + late temporal self-attention across frames
  C:            ViT + late temporal self-attention + perceiver compression
  B-per-stream: Per-camera late temporal self-attention
  C-per-stream: Per-camera late temporal self-attention + perceiver
  D:            Video ViT with CLS mid-fusion temporal attention
  E:            Video ViT with divided space-time attention

Uses timm for the ViT backbone with pretrained weight support.
Handles arbitrary image sizes via automatic pos_embed interpolation.
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def _load_vit_weights_from_file(model, weights_path, img_size, patch_size):
    """
    Load ViT weights from a local .pth file, with automatic
    positional embedding interpolation for different image sizes.

    Args:
        model: timm ViT model (already created with correct img_size)
        weights_path: path to .pth file containing state_dict
        img_size: target image size (after crop)
        patch_size: patch size used by the model
    """
    state_dict = torch.load(weights_path, map_location='cpu')

    # Handle common wrapper keys (e.g. 'model', 'state_dict')
    if 'model' in state_dict and isinstance(state_dict['model'], dict):
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        state_dict = state_dict['state_dict']

    # Interpolate pos_embed if sizes don't match
    if 'pos_embed' in state_dict:
        pos_embed_ckpt = state_dict['pos_embed']  # (1, N_ckpt, D)
        pos_embed_model = model.pos_embed  # (1, N_model, D)

        if pos_embed_ckpt.shape != pos_embed_model.shape:
            D = pos_embed_ckpt.shape[-1]
            # Separate CLS token (first) from spatial tokens
            cls_token = pos_embed_ckpt[:, :1, :]  # (1, 1, D)
            spatial_ckpt = pos_embed_ckpt[:, 1:, :]  # (1, N_spatial_ckpt, D)

            # Infer source grid size
            n_spatial_ckpt = spatial_ckpt.shape[1]
            gs_ckpt = int(math.sqrt(n_spatial_ckpt))
            assert gs_ckpt * gs_ckpt == n_spatial_ckpt, \
                f"Cannot infer grid from {n_spatial_ckpt} spatial tokens"

            # Target grid size
            gs_target = img_size // patch_size

            # Reshape to 2D grid and bicubic interpolate
            # Match timm's interpolation: float32 + antialias=True
            orig_dtype = spatial_ckpt.dtype
            spatial_ckpt = spatial_ckpt.float()
            spatial_ckpt = spatial_ckpt.reshape(1, gs_ckpt, gs_ckpt, D)
            spatial_ckpt = spatial_ckpt.permute(0, 3, 1, 2)  # (1, D, gs, gs)
            spatial_interp = F.interpolate(
                spatial_ckpt, size=(gs_target, gs_target),
                mode='bicubic', antialias=True)
            spatial_interp = spatial_interp.permute(
                0, 2, 3, 1).reshape(1, gs_target * gs_target, D)
            spatial_interp = spatial_interp.to(orig_dtype)

            state_dict['pos_embed'] = torch.cat(
                [cls_token, spatial_interp], dim=1)
            print(f"  pos_embed interpolated: ({gs_ckpt}x{gs_ckpt}) -> "
                  f"({gs_target}x{gs_target})")

    # Remove head keys that won't match (num_classes=0)
    keys_to_remove = [k for k in state_dict if k.startswith('head.')]
    for k in keys_to_remove:
        del state_dict[k]

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")


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


class PerceiverCompressor(nn.Module):
    """
    Compress variable-length token sequence into K fixed learned queries
    via iterative cross-attention.
    """

    def __init__(self, dim, num_queries, depth=2, num_heads=6,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.zeros(1, num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'q_norm': nn.LayerNorm(dim),
                'kv_norm': nn.LayerNorm(dim),
                'cross_attn': nn.MultiheadAttention(
                    dim, num_heads, dropout=dropout, batch_first=True),
                'ff_norm': nn.LayerNorm(dim),
                'ff': nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(int(dim * mlp_ratio), dim),
                    nn.Dropout(dropout),
                ),
            }))

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: (B, T, D) input tokens
            key_padding_mask: (B, T) True for PADDED positions
        Returns: (B, K, D)
        """
        B = x.shape[0]
        queries = self.queries.expand(B, -1, -1)

        for layer in self.layers:
            q = layer['q_norm'](queries)
            kv = layer['kv_norm'](x)
            queries = queries + layer['cross_attn'](
                q, kv, kv, key_padding_mask=key_padding_mask)[0]
            queries = queries + layer['ff'](layer['ff_norm'](queries))

        return queries


class VideoViTEncoder(nn.Module):
    """
    ViT with interleaved temporal attention for video encoding.

    Takes a pretrained timm ViT and inserts temporal self-attention every
    `temporal_every_k` spatial blocks. Two modes:

      "cls_only" (variant D):
        Temporal attention only between CLS tokens of different frames.
        Cheap — adds <1% compute. CLS carries temporal info into later
        spatial layers.

      "divided_st" (variant E):
        Temporal attention across ALL token positions independently
        (TimeSformer "Divided Space-Time" style). Every patch position
        attends to its counterpart across frames. More expressive but
        ~5-10% more compute.

    Spatial ViT blocks use pretrained weights. Temporal layers are randomly
    initialized (standard practice for video transformers).
    """

    def __init__(self, vit_model, temporal_mode, temporal_every_k=3,
                 max_frames=100, temporal_num_heads=6,
                 temporal_mlp_ratio=4.0, temporal_dropout=0.0):
        super().__init__()
        assert temporal_mode in ("cls_only", "divided_st")
        self.temporal_mode = temporal_mode
        self.temporal_every_k = temporal_every_k

        # Steal submodules from the timm ViT
        self.patch_embed = vit_model.patch_embed
        self.cls_token = vit_model.cls_token
        self.pos_embed = vit_model.pos_embed
        self.pos_drop = getattr(vit_model, 'pos_drop', nn.Identity())
        self.norm = vit_model.norm
        self.embed_dim = vit_model.embed_dim

        # Convert blocks to a ModuleList so we can iterate
        self.blocks = nn.ModuleList(vit_model.blocks)
        self.num_blocks = len(self.blocks)

        # Temporal position embedding (shared across insertion points)
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_frames, self.embed_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

        # One temporal attention layer per insertion point
        self.temporal_layers = nn.ModuleList()
        self._temporal_indices = []
        for i in range(self.num_blocks):
            if (i + 1) % temporal_every_k == 0:
                self.temporal_layers.append(
                    TemporalTransformerBlock(
                        self.embed_dim, temporal_num_heads,
                        mlp_ratio=temporal_mlp_ratio,
                        dropout=temporal_dropout))
                self._temporal_indices.append(i)
            else:
                self.temporal_layers.append(None)

        n_temporal = len(self._temporal_indices)
        print(f"  VideoViT ({temporal_mode}): {n_temporal} temporal layers "
              f"at blocks {self._temporal_indices}")

    def forward(self, x, T, key_padding_mask=None):
        """
        Args:
            x: (B*T, C, H, W) all frames batched together
            T: number of frames per sample
            key_padding_mask: (B, T) True for PADDED frames (optional)
        Returns:
            (B*T, embed_dim) CLS token per frame
        """
        BT = x.shape[0]
        B = BT // T

        # Patch embed + add CLS + spatial pos embed
        x = self.patch_embed(x)                     # (B*T, N, D)
        cls = self.cls_token.expand(BT, -1, -1)     # (B*T, 1, D)
        x = torch.cat([cls, x], dim=1)              # (B*T, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        N_plus_1 = x.shape[1]  # e.g. 50

        for i in range(self.num_blocks):
            # Standard spatial attention (batched over B*T)
            x = self.blocks[i](x)

            # Temporal attention at insertion points
            if self.temporal_layers[i] is not None:
                x = self._apply_temporal(
                    x, B, T, N_plus_1, self.temporal_layers[i],
                    key_padding_mask)

        x = self.norm(x)
        return x[:, 0]  # CLS tokens: (B*T, D)

    def _apply_temporal(self, x, B, T, N, temp_layer, key_padding_mask):
        """Apply temporal attention at a single insertion point."""
        D = x.shape[-1]

        if self.temporal_mode == "cls_only":
            # Temporal attention only between CLS tokens
            cls = x[:, 0, :].reshape(B, T, D)            # (B, T, D)
            cls = cls + self.temporal_pos_embed[:, :T, :]
            cls = temp_layer(cls, key_padding_mask=key_padding_mask)
            # Write back — clone to avoid in-place on autograd graph
            x = x.clone()
            x[:, 0, :] = cls.reshape(B * T, D)

        elif self.temporal_mode == "divided_st":
            # Temporal attention for ALL spatial positions
            x = x.reshape(B, T, N, D)
            x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # (B*N, T, D)
            x = x + self.temporal_pos_embed[:, :T, :]

            # Expand padding mask for all spatial positions
            if key_padding_mask is not None:
                kpm = key_padding_mask.unsqueeze(1).expand(
                    -1, N, -1).reshape(B * N, T)
            else:
                kpm = None

            x = temp_layer(x, key_padding_mask=kpm)
            x = x.reshape(B, N, T, D).permute(0, 2, 1, 3)   # (B, T, N, D)
            x = x.reshape(B * T, N, D)

        return x


class ViTObsEncoder(ModuleAttrMixin):
    """
    ViT-based observation encoder. Reads shape_meta to discover obs keys
    and their shapes — nothing is hardcoded.

    Each RGB key gets its own ViT encoder (or shared) with CropRandomizer.
    Low-dim keys are concatenated as-is.
    """

    VALID_VARIANTS = ("A-small", "A-full", "B", "C", "B-per-stream", "C-per-stream", "D", "E")

    def __init__(self,
                 shape_meta: dict,
                 variant: str = "A-small",
                 # ViT backbone
                 vit_model_name: str = "vit_small_patch16_224",
                 pretrained=True,  # bool (download) or str (local path)
                 vit_freeze: bool = False,
                 # Projection (A-small only)
                 projection_dim: int = 64,
                 # Crop augmentation
                 crop_shape: tuple = (112, 112),
                 # Normalization: exactly one must be True
                 imagenet_norm: bool = False,
                 # Temporal encoder (B, C)
                 temporal_depth: int = 4,
                 temporal_num_heads: int = 6,
                 max_frames: int = 100,
                 temporal_mlp_ratio: float = 4.0,
                 temporal_dropout: float = 0.0,
                 # Perceiver (C)
                 num_perceiver_queries: int = 32,
                 perceiver_depth: int = 2,
                 # Camera weight sharing
                 share_rgb_model: bool = False,
                 # Video ViT (D, E)
                 temporal_every_k: int = 3,
                 ):
        super().__init__()
        assert variant in self.VALID_VARIANTS, \
            f"variant must be one of {self.VALID_VARIANTS}, got '{variant}'"
        self.variant = variant

        # --- Parse shape_meta ---
        rgb_keys = []
        low_dim_keys = []
        key_shape_map = {}

        for key, attr in shape_meta['obs'].items():
            shape = tuple(attr['shape'])
            key_shape_map[key] = shape
            type_ = attr.get('type', 'low_dim')
            if type_ == 'rgb':
                rgb_keys.append(key)
            elif type_ == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type_}")

        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        # --- Normalization failsafe ---
        # ImageNet norm is applied inside this encoder.
        # If imagenet_norm=False, the caller (policy) MUST apply its own
        # normalization (e.g. LinearNormalizer) to RGB inputs before calling
        # this encoder.  Both cannot be active at the same time.
        self.use_imagenet_norm = imagenet_norm
        if imagenet_norm:
            self.register_buffer(
                'img_mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer(
                'img_std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # --- Crop randomizers ---
        self.crop_randomizers = nn.ModuleDict()
        for key in rgb_keys:
            img_shape = key_shape_map[key]  # (C, H, W)
            ch, cw = crop_shape
            if ch < img_shape[1] and cw < img_shape[2]:
                self.crop_randomizers[key] = CropRandomizer(
                    input_shape=img_shape,
                    crop_height=ch,
                    crop_width=cw,
                    num_crops=1,
                    pos_enc=False,
                )

        # --- ViT encoders ---
        # pretrained can be:
        #   False  -> random init
        #   True   -> download pretrained weights (cached after first use)
        #   str    -> load from local .pth file (for offline/SLURM nodes)
        self.share_rgb_model = share_rgb_model
        self.vit_encoders = nn.ModuleDict()

        pretrained_is_path = isinstance(pretrained, str)
        timm_pretrained = pretrained if isinstance(pretrained, bool) else False

        # Infer patch_size from model name for pos_embed interpolation
        # e.g. "vit_small_patch16_224" -> 16
        _patch_size = 16  # default
        for part in vit_model_name.split('_'):
            if part.startswith('patch'):
                _patch_size = int(part.replace('patch', ''))

        def _create_vit():
            vit = timm.create_model(
                vit_model_name, pretrained=timm_pretrained,
                img_size=crop_shape[0], num_classes=0)
            if pretrained_is_path:
                print(f"Loading ViT weights from: {pretrained}")
                _load_vit_weights_from_file(
                    vit, pretrained, crop_shape[0], _patch_size)
            return vit

        # For D/E, wrap each ViT in a VideoViTEncoder
        _is_video_vit = variant in ("D", "E")
        _temporal_mode = {"D": "cls_only", "E": "divided_st"}.get(variant)

        if share_rgb_model:
            vit = _create_vit()
            self._vit_embed_dim = vit.embed_dim
            if _is_video_vit:
                vit = VideoViTEncoder(
                    vit, temporal_mode=_temporal_mode,
                    temporal_every_k=temporal_every_k,
                    max_frames=max_frames,
                    temporal_num_heads=temporal_num_heads,
                    temporal_mlp_ratio=temporal_mlp_ratio,
                    temporal_dropout=temporal_dropout)
            for key in rgb_keys:
                self.vit_encoders[key] = vit  # same reference
        else:
            base_vit = _create_vit()
            self._vit_embed_dim = base_vit.embed_dim
            if _is_video_vit:
                base_vit = VideoViTEncoder(
                    base_vit, temporal_mode=_temporal_mode,
                    temporal_every_k=temporal_every_k,
                    max_frames=max_frames,
                    temporal_num_heads=temporal_num_heads,
                    temporal_mlp_ratio=temporal_mlp_ratio,
                    temporal_dropout=temporal_dropout)
            self.vit_encoders[rgb_keys[0]] = base_vit
            for key in rgb_keys[1:]:
                self.vit_encoders[key] = copy.deepcopy(base_vit)

        # Freeze ViT if requested
        if vit_freeze:
            for key in rgb_keys:
                for param in self.vit_encoders[key].parameters():
                    param.requires_grad = False

        # --- Per-camera projection (A-small) ---
        self.projectors = nn.ModuleDict()
        if variant == "A-small":
            per_camera_dim = projection_dim
            for key in rgb_keys:
                self.projectors[key] = nn.Linear(
                    self._vit_embed_dim, projection_dim)
        else:
            per_camera_dim = self._vit_embed_dim

        # --- Compute output dim ---
        low_dim_total = sum(key_shape_map[k][-1] for k in low_dim_keys)
        self._output_dim = per_camera_dim * len(rgb_keys) + low_dim_total
        self._per_camera_dim = per_camera_dim

        # --- Temporal encoder (B, C) ---
        # The temporal encoder and perceiver operate in a clean internal
        # dimension (temporal_dim) that is divisible by num_heads.
        # We project output_dim <-> temporal_dim at the boundaries.
        self.temporal_encoder = None
        self.temporal_proj_in = None
        self.temporal_proj_out = None
        self._num_perceiver_queries = num_perceiver_queries
        self._per_stream = variant in ("B-per-stream", "C-per-stream")

        if variant in ("B", "C"):
            temporal_dim = temporal_num_heads * (
                self._output_dim // temporal_num_heads
                + (1 if self._output_dim % temporal_num_heads else 0))
            # If output_dim is already divisible, no projection needed
            if temporal_dim != self._output_dim:
                self.temporal_proj_in = nn.Linear(
                    self._output_dim, temporal_dim)
                self.temporal_proj_out = nn.Linear(
                    temporal_dim, self._output_dim)
            else:
                temporal_dim = self._output_dim

            self.temporal_encoder = TemporalEncoder(
                dim=temporal_dim,
                depth=temporal_depth,
                num_heads=temporal_num_heads,
                max_frames=max_frames,
                mlp_ratio=temporal_mlp_ratio,
                dropout=temporal_dropout,
            )

        # --- Per-stream temporal encoders (B-per-stream, C-per-stream) ---
        # Each RGB camera gets its own temporal encoder operating on
        # per_camera_dim. Low-dim streams are too small for multi-head
        # attention and are concatenated without temporal processing.
        self.per_stream_temporal_encoders = nn.ModuleDict()
        if self._per_stream:
            for key in rgb_keys:
                self.per_stream_temporal_encoders[key] = TemporalEncoder(
                    dim=per_camera_dim,
                    depth=temporal_depth,
                    num_heads=temporal_num_heads,
                    max_frames=max_frames,
                    mlp_ratio=temporal_mlp_ratio,
                    dropout=temporal_dropout,
                )

        # --- Perceiver (C, C-per-stream) ---
        # Perceiver always runs on the full concatenated output_dim,
        # after per-stream temporal attention has been applied.
        self.perceiver = None
        if variant in ("C", "C-per-stream"):
            if variant == "C":
                perceiver_dim = temporal_dim
            else:
                # For per-stream, perceiver operates on output_dim
                perceiver_dim = temporal_num_heads * (
                    self._output_dim // temporal_num_heads
                    + (1 if self._output_dim % temporal_num_heads else 0))
                if perceiver_dim != self._output_dim:
                    self.perceiver_proj_in = nn.Linear(
                        self._output_dim, perceiver_dim)
                    self.perceiver_proj_out = nn.Linear(
                        perceiver_dim, self._output_dim)
                else:
                    self.perceiver_proj_in = None
                    self.perceiver_proj_out = None
            self.perceiver = PerceiverCompressor(
                dim=perceiver_dim,
                num_queries=num_perceiver_queries,
                depth=perceiver_depth,
                num_heads=temporal_num_heads,
                mlp_ratio=temporal_mlp_ratio,
                dropout=temporal_dropout,
            )

        # --- Print summary ---
        vit_params = sum(
            p.numel() for k in rgb_keys
            for p in self.vit_encoders[k].parameters())
        if share_rgb_model:
            vit_params = sum(
                p.numel() for p in self.vit_encoders[rgb_keys[0]].parameters())
        print(f"ViT encoder variant={variant}, output_dim={self._output_dim}")
        print(f"  ViT params: {vit_params:,}")
        if self.temporal_encoder is not None:
            t_params = sum(
                p.numel() for p in self.temporal_encoder.parameters())
            print(f"  Temporal encoder params: {t_params:,}")
        if self.per_stream_temporal_encoders:
            for key in rgb_keys:
                t_params = sum(
                    p.numel()
                    for p in self.per_stream_temporal_encoders[key].parameters())
                print(f"  Per-stream temporal ({key}) params: {t_params:,}")
        if self.perceiver is not None:
            p_params = sum(
                p.numel() for p in self.perceiver.parameters())
            print(f"  Perceiver params: {p_params:,}")

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def needs_temporal(self):
        return self.variant in ("B", "C", "B-per-stream", "C-per-stream")

    @property
    def is_video_vit(self):
        return self.variant in ("D", "E")

    @property
    def num_perceiver_queries(self):
        return self._num_perceiver_queries

    def output_shape(self):
        """Match the interface of the RoboMimic encoder."""
        return (self._output_dim,)

    def variant_output_tokens(self, n_obs_steps):
        """Number of tokens this encoder outputs for the UNet cross-attention."""
        if self.variant in ("C", "C-per-stream"):
            return self._num_perceiver_queries
        return n_obs_steps

    def _encode_per_frame(self, obs_dict):
        """
        Per-frame ViT encoding. Returns per-key features (not concatenated).

        Args:
            obs_dict: {rgb_key: (N, C, H, W), low_dim_key: (N, D)}
                where N is a flat batch (typically B*T).
        Returns: dict of {key: (N, dim)} features
        """
        features = {}

        for key in self.rgb_keys:
            img = obs_dict[key]  # (N, C, H, W)

            # Crop augmentation
            if key in self.crop_randomizers:
                img = self.crop_randomizers[key](img)

            # ImageNet normalization
            if self.use_imagenet_norm:
                img = (img - self.img_mean) / self.img_std

            # ViT forward — returns CLS token
            cls = self.vit_encoders[key](img)  # (N, vit_embed_dim)

            # Project if A-small
            if self.variant == "A-small":
                cls = self.projectors[key](cls)

            features[key] = cls

        for key in self.low_dim_keys:
            features[key] = obs_dict[key]

        return features

    def forward(self, obs_dict):
        """
        Per-frame encoding. Used for all variants.

        Args:
            obs_dict: {rgb_key: (N, C, H, W), low_dim_key: (N, D)}
                where N is a flat batch (typically B*T).
        Returns: (N, output_dim)
        """
        features = self._encode_per_frame(obs_dict)
        # Concatenate in sorted key order (rgb first, then low_dim)
        parts = [features[k] for k in self.rgb_keys] + \
                [features[k] for k in self.low_dim_keys]
        return torch.cat(parts, dim=-1)  # (N, output_dim)

    def forward_video(self, obs_dict, T, key_padding_mask=None):
        """
        Video encoding for variants D and E. Passes frame count T to the
        VideoViTEncoder so it can apply temporal attention within ViT blocks.

        Args:
            obs_dict: {rgb_key: (B*T, C, H, W), low_dim_key: (B*T, D)}
                where B*T includes all valid frames (already selected).
            T: number of frames per sample (int)
            key_padding_mask: (B, T) True for PADDED frames (optional)
        Returns: (B*T, output_dim) per-frame features with temporal info baked in
        """
        assert self.variant in ("D", "E"), \
            f"forward_video only for variants D/E, got '{self.variant}'"

        features = {}
        for key in self.rgb_keys:
            img = obs_dict[key]  # (B*T, C, H, W)

            if key in self.crop_randomizers:
                img = self.crop_randomizers[key](img)

            if self.use_imagenet_norm:
                img = (img - self.img_mean) / self.img_std

            # VideoViTEncoder.forward expects (B*T, C, H, W) and T
            cls = self.vit_encoders[key](img, T,
                                         key_padding_mask=key_padding_mask)

            if self.variant == "A-small":
                cls = self.projectors[key](cls)

            features[key] = cls

        for key in self.low_dim_keys:
            features[key] = obs_dict[key]

        parts = [features[k] for k in self.rgb_keys] + \
                [features[k] for k in self.low_dim_keys]
        return torch.cat(parts, dim=-1)


        """
        Temporal processing for variants B and C (early fusion).
        Called AFTER forward() results are reshaped to (B, T, D).

        Args:
            x: (B, T, output_dim) per-timestep features
            key_padding_mask: (B, T) True for PADDED positions
        Returns:
            variant B: (B, T, output_dim)
            variant C: (B, K, output_dim)
        """
        assert self.needs_temporal and not self._per_stream, \
            "forward_temporal only for variants B/C (not per-stream)"

        if self.temporal_proj_in is not None:
            x = self.temporal_proj_in(x)

        x = self.temporal_encoder(x, key_padding_mask=key_padding_mask)

        if self.variant == "C":
            x = self.perceiver(x, key_padding_mask=key_padding_mask)

        if self.temporal_proj_out is not None:
            x = self.temporal_proj_out(x)

        return x

    def forward_per_stream_temporal(self, per_key_features, key_padding_mask=None):
        """
        Per-stream temporal processing for B-per-stream and C-per-stream.

        Applies independent temporal attention to each RGB camera stream,
        then concatenates with low-dim features before optional perceiver.

        Args:
            per_key_features: dict of {key: (B, T, dim)} per-stream features
            key_padding_mask: (B, T) True for PADDED positions
        Returns:
            B-per-stream: (B, T, output_dim)
            C-per-stream: (B, K, output_dim)
        """
        assert self._per_stream, \
            "forward_per_stream_temporal only for per-stream variants"

        parts = []
        # Apply temporal attention independently to each RGB stream
        for key in self.rgb_keys:
            stream = per_key_features[key]  # (B, T, per_camera_dim)
            stream = self.per_stream_temporal_encoders[key](
                stream, key_padding_mask=key_padding_mask)
            parts.append(stream)

        # Low-dim features pass through without temporal attention
        for key in self.low_dim_keys:
            parts.append(per_key_features[key])

        x = torch.cat(parts, dim=-1)  # (B, T, output_dim)

        # Perceiver compression (C-per-stream only)
        if self.variant == "C-per-stream":
            if self.perceiver_proj_in is not None:
                x = self.perceiver_proj_in(x)
            x = self.perceiver(x, key_padding_mask=key_padding_mask)
            if self.perceiver_proj_out is not None:
                x = self.perceiver_proj_out(x)

        return x
