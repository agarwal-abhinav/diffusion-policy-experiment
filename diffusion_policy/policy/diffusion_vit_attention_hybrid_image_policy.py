"""
Diffusion policy with ViT-based observation encoder and attention-based
conditioning in the UNet. Supports 4 ViT encoder variants (A-small,
A-full, B, C) for long-context imitation learning experiments.

Based on DiffusionAttentionHybridImagePolicy but replaces the RoboMimic
ResNet18+SpatialSoftmax encoder with a ViT encoder from timm.
"""

from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.attention_conditional_unet1d import AttentionConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.vit_obs_encoder import ViTObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionViTAttentionHybridImagePolicy(BaseImagePolicy):
    """
    Attention-based diffusion policy with ViT observation encoder.
    Supports variable-length observation sequences and 4 ViT variants.
    """

    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon,
                 n_action_steps,
                 n_obs_steps,
                 # ViT encoder config (passed to ViTObsEncoder)
                 vit_variant="A-small",
                 vit_model_name="vit_small_patch16_224",
                 vit_pretrained=True,
                 vit_freeze=False,
                 vit_projection_dim=64,
                 crop_shape=(112, 112),
                 imagenet_norm=False,
                 share_rgb_model=False,
                 # Temporal / Perceiver
                 temporal_depth=4,
                 temporal_num_heads=6,
                 max_frames=100,
                 temporal_mlp_ratio=4.0,
                 temporal_dropout=0.0,
                 num_perceiver_queries=32,
                 perceiver_depth=2,
                 # Video ViT (D, E)
                 temporal_every_k=3,
                 # UNet config
                 diffusion_step_embed_dim=256,
                 down_dims=(256, 512, 1024),
                 kernel_size=5,
                 n_groups=8,
                 num_attention_heads=8,
                 attention_dropout=0.1,
                 # Diffusion
                 num_inference_steps=None,
                 num_DDIM_inference_steps=10,
                 # Optional features
                 one_hot_encoding_dim=0,
                 use_target_cond=False,
                 target_dim=None,
                 obs_as_global_cond=True,
                 # Prediction horizon control
                 n_past_action_steps=None,
                 # kwargs passed to scheduler.step
                 **kwargs):
        super().__init__()

        if use_target_cond:
            assert target_dim is not None
        assert one_hot_encoding_dim >= 0
        assert obs_as_global_cond

        # --- Parse action dim ---
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # --- Build ViT encoder ---
        obs_encoder = ViTObsEncoder(
            shape_meta=shape_meta,
            variant=vit_variant,
            vit_model_name=vit_model_name,
            pretrained=vit_pretrained,
            vit_freeze=vit_freeze,
            projection_dim=vit_projection_dim,
            crop_shape=crop_shape,
            imagenet_norm=imagenet_norm,
            temporal_depth=temporal_depth,
            temporal_num_heads=temporal_num_heads,
            max_frames=max_frames,
            temporal_mlp_ratio=temporal_mlp_ratio,
            temporal_dropout=temporal_dropout,
            num_perceiver_queries=num_perceiver_queries,
            perceiver_depth=perceiver_depth,
            share_rgb_model=share_rgb_model,
            temporal_every_k=temporal_every_k,
        )

        obs_feature_dim = obs_encoder.output_dim
        max_global_tokens = obs_encoder.variant_output_tokens(n_obs_steps)

        print(f"Input dim: {action_dim}, Global cond dim: {obs_feature_dim}")
        print(f"Max global tokens: {max_global_tokens}")

        # --- Build UNet ---
        model = AttentionConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim,
            target_dim=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            max_global_tokens=max_global_tokens,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler

        # DDIM scheduler
        DDIM_noise_scheduler = DDIMScheduler(
            num_train_timesteps=noise_scheduler.num_train_timesteps,
            beta_start=noise_scheduler.beta_start,
            beta_end=noise_scheduler.beta_end,
            beta_schedule=noise_scheduler.beta_schedule,
            clip_sample=noise_scheduler.clip_sample,
            prediction_type=noise_scheduler.prediction_type,
        )
        DDIM_noise_scheduler.set_timesteps(num_DDIM_inference_steps)
        self.DDIM_noise_scheduler = DDIM_noise_scheduler
        self.num_DDIM_inference_steps = num_DDIM_inference_steps

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        # When imagenet_norm is active inside the encoder, we must NOT
        # apply LinearNormalizer to RGB keys (it would double-normalize).
        self.skip_rgb_linear_norm = obs_encoder.use_imagenet_norm
        self.rgb_keys = obs_encoder.rgb_keys  # sorted list of RGB key names
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.max_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.one_hot_encoding_dim = one_hot_encoding_dim
        self.use_target_cond = use_target_cond
        self.max_global_tokens = max_global_tokens
        self.kwargs = kwargs

        # --- Prediction horizon ---
        # n_past_action_steps=None means full horizon (backward compatible).
        # Otherwise, the diffusion model denoises a shorter sequence whose
        # length depends on the per-sample context length To:
        #   pred_len = min(n_past_action_steps, To) + n_future
        # UNet requires sequence length divisible by 4 (2 downsampling stages)
        self._unet_alignment = 4
        self.n_past_action_steps = n_past_action_steps
        self.n_future = horizon - n_obs_steps

        if n_past_action_steps is not None:
            # Minimum prediction horizon (when To >= n_past)
            min_prediction_horizon = n_past_action_steps + self.n_future
            assert min_prediction_horizon >= n_action_steps, \
                f"min prediction_horizon ({min_prediction_horizon}) must be >= " \
                f"n_action_steps ({n_action_steps})"
            # Static prediction horizon used when context is fixed
            self.prediction_horizon = min_prediction_horizon
            self._action_offset = horizon - min_prediction_horizon
            print(f"Prediction horizon (static): {min_prediction_horizon} "
                  f"(past={n_past_action_steps}, future={self.n_future}, "
                  f"offset={self._action_offset})")
        else:
            self.prediction_horizon = horizon
            self._action_offset = 0

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(
            p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(
            p.numel() for p in self.obs_encoder.parameters()))

    # ========= UNet alignment helpers ============

    def _pad_to_unet(self, x):
        """Pad sequence dim to be divisible by UNet alignment. Returns (padded, orig_len)."""
        T = x.shape[1]
        remainder = T % self._unet_alignment
        if remainder == 0:
            return x, T
        pad_len = self._unet_alignment - remainder
        # Zero-pad at the end of the sequence dimension
        x = F.pad(x, (0, 0, 0, pad_len))  # pad last 2 dims: (dim, dim, seq, seq)
        return x, T

    def _unpad_from_unet(self, x, orig_len):
        """Remove padding added by _pad_to_unet."""
        return x[:, :orig_len, :]

    # ========= normalization ============

    def _normalize_obs(self, obs_dict):
        """
        Normalize observations. If imagenet_norm is active in the encoder,
        skip LinearNormalizer for RGB keys (encoder handles them).
        """
        if not self.skip_rgb_linear_norm:
            return self.normalizer.normalize(obs_dict)
        # Selective normalization: only normalize non-RGB keys
        result = {}
        for key, val in obs_dict.items():
            if key in self.rgb_keys:
                result[key] = val  # raw — encoder will apply ImageNet norm
            else:
                result[key] = self.normalizer[key].normalize(val)
        return result

    # ========= observation encoding ============

    def _prepare_batch_and_apply_obs_encoding(self, nobs, num_obs_steps_in_this):
        """
        Encode variable-length observations with ViT.

        For variants A: per-frame ViT encoding, same as baseline.
        For variants B/C: per-frame encoding + temporal attention
            (+ perceiver for C).

        Args:
            nobs: dict of normalized obs, each (B, N, ...)
            num_obs_steps_in_this: (B,) per-sample obs count
        Returns:
            embeddings: (B, T_out, obs_feature_dim)
            global_mask: (B, T_out) True for valid positions
            position_mask: (B, T_out) right-aligned position indices
        """
        B = nobs[next(iter(nobs))].shape[0]
        N = nobs[next(iter(nobs))].shape[1]

        K = num_obs_steps_in_this.unsqueeze(1)  # (B, 1)

        arrange_N = torch.arange(N, device=self.device)
        left_mask = arrange_N.unsqueeze(0) < K  # (B, N)

        # Select valid frames and encode
        x_selected = dict_apply(nobs, lambda x: x[left_mask])

        # --- Video ViT path (D, E) ---
        if self.obs_encoder.is_video_vit:
            # VideoViTEncoder needs constant T per batch with padding mask.
            # Trim to n_obs_steps to avoid processing NaN-padded frames.
            T = self.n_obs_steps
            nobs_trimmed = dict_apply(nobs, lambda x: x[:, :T])
            x_flat = dict_apply(nobs_trimmed, lambda x: x.reshape(-1, *x.shape[2:]))
            padding_mask = ~left_mask[:, :T]  # (B, T), True for padded
            embeddings = self.obs_encoder.forward_video(
                x_flat, T, key_padding_mask=padding_mask)  # (B*T, D)
            embeddings_full = embeddings.reshape(B, T, -1)
            global_mask = left_mask[:, :T]

        # --- Per-stream temporal path (B-per-stream, C-per-stream) ---
        elif self.obs_encoder._per_stream:
            # Get per-key features before concatenation
            per_key_features = self.obs_encoder._encode_per_frame(x_selected)

            # Scatter each key back to (B, N, dim)
            per_key_full = {}
            for key, feat in per_key_features.items():
                dim = feat.shape[-1]
                full = feat.new_zeros((B, N, dim))
                full[left_mask] = feat
                per_key_full[key] = full[:, :self.n_obs_steps, :]

            global_mask = left_mask[:, :self.n_obs_steps]
            padding_mask = ~global_mask

            embeddings_full = self.obs_encoder.forward_per_stream_temporal(
                per_key_full, key_padding_mask=padding_mask)

            if self.obs_encoder.variant == "C-per-stream":
                K_p = self.obs_encoder.num_perceiver_queries
                global_mask = torch.ones(
                    B, K_p, dtype=torch.bool, device=self.device)
                position_mask = torch.arange(
                    K_p, device=self.device).unsqueeze(0).expand(B, -1)
                return embeddings_full, global_mask, position_mask

        else:
            # Standard path: encode and concatenate
            embeddings = self.obs_encoder(x_selected)

            # Scatter back to (B, N, D)
            embeddings_full = embeddings.new_zeros(
                (B, N, self.obs_feature_dim))
            embeddings_full[left_mask] = embeddings
            embeddings_full = embeddings_full[:, :self.n_obs_steps, :]

            global_mask = left_mask[:, :self.n_obs_steps]

            # --- Temporal path for B/C (early fusion) ---
            if self.obs_encoder.needs_temporal:
                padding_mask = ~global_mask
                embeddings_full = self.obs_encoder.forward_temporal(
                    embeddings_full, key_padding_mask=padding_mask)

                if self.obs_encoder.variant == "C":
                    K_p = self.obs_encoder.num_perceiver_queries
                    global_mask = torch.ones(
                        B, K_p, dtype=torch.bool, device=self.device)
                    position_mask = torch.arange(
                        K_p, device=self.device).unsqueeze(0).expand(B, -1)
                    return embeddings_full, global_mask, position_mask

        # --- Right-aligned position encoding (A, B) ---
        idx = torch.arange(
            self.n_obs_steps, device=self.device).unsqueeze(0)
        first_k = idx < K
        position_mask = torch.where(
            first_k, idx + (self.n_obs_steps - K),
            torch.zeros_like(idx))

        return embeddings_full, global_mask, position_mask

    # ========= inference ============

    def conditional_sample(self,
                           condition_data, condition_mask,
                           local_cond=None, global_cond=None,
                           global_mask=None, target_cond=None,
                           temporal_positions=None, generator=None,
                           use_DDIM=False,
                           **kwargs):
        model = self.model
        if use_DDIM:
            scheduler = self.DDIM_noise_scheduler
            scheduler.set_timesteps(self.num_DDIM_inference_steps)
        else:
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(self.num_inference_steps)

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # Pad for UNet alignment if needed
        trajectory, orig_len = self._pad_to_unet(trajectory)
        condition_data_pad, _ = self._pad_to_unet(condition_data)
        condition_mask_pad, _ = self._pad_to_unet(condition_mask.float())
        condition_mask_pad = condition_mask_pad.bool()

        for t in scheduler.timesteps:
            trajectory[condition_mask_pad] = condition_data_pad[condition_mask_pad]

            model_output = model(
                trajectory, t,
                global_cond=global_cond, global_mask=global_mask,
                target_cond=target_cond, local_cond=local_cond,
                temporal_positions=temporal_positions)

            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs).prev_sample

        trajectory[condition_mask_pad] = condition_data_pad[condition_mask_pad]
        trajectory = self._unpad_from_unet(trajectory, orig_len)
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       num_obs_tokens: Optional[int] = None,
                       use_DDIM=False,
                       use_custom_inference_tokens=None
                       ) -> Dict[str, torch.Tensor]:
        assert 'obs' in obs_dict
        if self.use_target_cond:
            assert 'target' in obs_dict
        assert 'past_action' not in obs_dict

        nobs = self._normalize_obs(obs_dict['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(obs_dict['target'])

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        if use_custom_inference_tokens is not None:
            To = min(To, use_custom_inference_tokens)
        Da = self.action_dim

        if num_obs_tokens is None:
            if use_custom_inference_tokens is not None:
                def shifting_tokens(tensor):
                    new_tensor = tensor.new_zeros(tensor.shape)
                    new_tensor[:, :use_custom_inference_tokens, ...] = \
                        tensor[:, -use_custom_inference_tokens:, ...]
                    return new_tensor
                nobs = dict_apply(nobs, shifting_tokens)
                num_obs_tokens = torch.ones(
                    B, dtype=torch.long,
                    device=value.device) * use_custom_inference_tokens
            else:
                num_obs_tokens = torch.ones(
                    B, dtype=torch.long,
                    device=value.device) * self.max_obs_steps

        # To should be the actual obs step count, not the tensor width
        # (tensor may be horizon-length with NaN padding beyond obs steps)
        To = num_obs_tokens[0].item()

        # Compute dynamic prediction horizon
        if self.n_past_action_steps is not None:
            past_in_pred = min(self.n_past_action_steps, To)
            T = past_in_pred + self.n_future
            T = math.ceil(T / self._unet_alignment) * self._unet_alignment
        else:
            T = self.prediction_horizon
            past_in_pred = To

        device = self.device
        dtype = self.dtype

        global_cond, global_mask, temporal_positions = \
            self._prepare_batch_and_apply_obs_encoding(
                nobs, num_obs_tokens)

        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = obs_dict['one_hot_encoding']
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(
                -1, global_cond.shape[1], -1)
            global_cond = torch.cat(
                [global_cond, one_hot_expanded], dim=-1)

        cond_data = torch.zeros(
            size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(B, -1)

        nsample = self.conditional_sample(
            cond_data, cond_mask,
            local_cond=None,
            global_cond=global_cond,
            global_mask=global_mask,
            target_cond=target_cond,
            temporal_positions=temporal_positions,
            use_DDIM=use_DDIM,
            **self.kwargs)

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Current action is at position past_in_pred - 1 in the prediction
        start = past_in_pred - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # action_pred starts at this offset in the full horizon
        action_pred_offset = To - past_in_pred

        result = {
            'action': action,
            'action_pred': action_pred,
            'action_pred_offset': action_pred_offset,
            'num_obs_tokens_used': num_obs_tokens,
        }
        return result

    # ========= training ============

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch, noisy_trajectory, timesteps):
        assert 'valid_mask' not in batch
        nobs = self._normalize_obs(batch['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        global_cond, global_mask, temporal_positions = \
            self._prepare_batch_and_apply_obs_encoding(
                nobs, batch['sample_metadata']['num_obs_steps'])

        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch['one_hot_encoding']
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(
                -1, global_cond.shape[1], -1)
            global_cond = torch.cat(
                [global_cond, one_hot_expanded], dim=-1)

        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(nactions.shape[0], -1)

        # Pad for UNet alignment if needed
        noisy_trajectory, orig_len = self._pad_to_unet(noisy_trajectory)
        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=None, global_cond=global_cond,
            global_mask=global_mask, target_cond=target_cond,
            temporal_positions=temporal_positions)
        return self._unpad_from_unet(pred, orig_len)

    def _build_dynamic_trajectory(self, nactions, num_obs_steps):
        """
        Build per-sample trajectory for dynamic prediction horizon.

        For each sample with context length To:
          pred_len = min(n_past_action_steps, To) + n_future
          offset   = To - min(n_past_action_steps, To)
          trajectory[i] = nactions[i, offset : offset + pred_len]

        Shorter samples are right-padded to the batch max (rounded to UNet alignment).
        Returns trajectory, loss_mask (True for valid positions).
        """
        B = nactions.shape[0]
        Da = nactions.shape[2]
        device = nactions.device

        # Per-sample computation
        To = num_obs_steps  # (B,)
        past_in_pred = torch.clamp(To, max=self.n_past_action_steps)  # min(n_past, To)
        pred_lens = past_in_pred + self.n_future  # (B,)
        offsets = To - past_in_pred  # (B,) = max(0, To - n_past)

        max_pred = pred_lens.max().item()
        padded_pred = math.ceil(max_pred / self._unet_alignment) * self._unet_alignment

        # Build padded trajectory and loss mask
        trajectory = nactions.new_zeros((B, padded_pred, Da))
        loss_mask = torch.zeros((B, padded_pred, Da), dtype=torch.bool, device=device)

        for i in range(B):
            pl = pred_lens[i].item()
            off = offsets[i].item()
            trajectory[i, :pl, :] = nactions[i, off:off + pl, :]
            loss_mask[i, :pl, :] = True

        return trajectory, loss_mask

    def compute_loss(self, batch):
        assert 'valid_mask' not in batch
        nobs = self._normalize_obs(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        batch_size = nactions.shape[0]

        num_obs_steps = batch['sample_metadata']['num_obs_steps']
        global_cond, global_mask, temporal_positions = \
            self._prepare_batch_and_apply_obs_encoding(nobs, num_obs_steps)

        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch['one_hot_encoding']
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(
                -1, global_cond.shape[1], -1)
            global_cond = torch.cat(
                [global_cond, one_hot_expanded], dim=-1)

        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1)

        # Build trajectory (per-sample dynamic or static)
        if self.n_past_action_steps is not None:
            trajectory, dynamic_loss_mask = self._build_dynamic_trajectory(
                nactions, num_obs_steps)
        else:
            trajectory = nactions
            dynamic_loss_mask = None

        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        loss_mask = ~condition_mask
        if dynamic_loss_mask is not None:
            loss_mask = loss_mask & dynamic_loss_mask

        # Pad for UNet alignment if needed
        noisy_trajectory, orig_len = self._pad_to_unet(noisy_trajectory)
        pred = self.model(
            noisy_trajectory, timesteps,
            local_cond=None, global_cond=global_cond,
            global_mask=global_mask, target_cond=target_cond,
            temporal_positions=temporal_positions)
        pred = self._unpad_from_unet(pred, orig_len)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss
