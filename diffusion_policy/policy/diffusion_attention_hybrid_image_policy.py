from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.mlp.mlp import MLP
from diffusion_policy.model.diffusion.attention_conditional_unet1d import AttentionConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmobsc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.vision.vit_obs_encoder import TemporalEncoder


class DiffusionAttentionHybridImagePolicy(BaseImagePolicy):
    """
    Attention-based diffusion policy that uses variable-length observation sequences.
    Uses AttentionConditionalUnet1D for token-based conditioning with positional encoding.
    """
    
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            one_hot_encoding_dim=0,
            num_inference_steps=None,
            obs_as_global_cond=True,
            use_target_cond=False,
            target_dim=None,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            num_attention_heads=8,  # NEW: attention heads
            attention_dropout=0.1,  # NEW: attention dropout
            obs_embedding_dim=None,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            num_DDIM_inference_steps=10,
            pretrained_encoder=False,
            freeze_pretrained_encoder=False,
            initialize_obs_encoder=None, 
            freeze_self_trained_obs_encoder=False, 
            inference_loading=False,
            n_past_action_steps=None,
            # Frame selection (sparse obs conditioning)
            obs_frame_selection=None,
            # Per-stream temporal attention on encoder features
            use_temporal_attention=False,
            temporal_depth=4,
            temporal_num_heads=8,
            temporal_max_frames=100,
            # parameters passed to step
            **kwargs):
        super().__init__()

        if use_target_cond:
            assert target_dim is not None
        assert one_hot_encoding_dim >= 0
        assert obs_as_global_cond

        # NEW: Variable observation parameters
        self.max_obs_steps = n_obs_steps

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        # each list contains the keys of the corresponding modality
        # ex. {low_dim: [agent_pos], rgb: [image], depth: [], scan: []}
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        # ex. {agent_pos: shape, image: shape}
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            typee = attr.get('type', 'low_dim')
            if typee == 'rgb':
                obs_config['rgb'].append(key)
            elif typee == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {typee}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph', 
            pretrained_encoder=pretrained_encoder,
            freeze_pretrained_encoder=freeze_pretrained_encoder)
                
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        # extract the image encoder
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmobsc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # # Handle observation embedding
        # if obs_embedding_dim is not None:
        #     obs_feature_dim = obs_embedding_dim
        #     self.obs_embedding_projector = MLP(obs_encoder.output_shape()[0], [], obs_feature_dim)
        #     self.obs_embedding_projector.to('cuda' if torch.cuda.is_available() else 'cpu')
        #     project_obs_embedding = True
        # else:
        #     obs_feature_dim = obs_encoder.output_shape()[0]
        #     project_obs_embedding = False

        # NEW: create attention-based diffusion model
        input_dim = action_dim
        # Global conditioning will be handled via attention tokens, not concatenation
        global_cond_dim = obs_encoder.output_shape()[0]
        obs_feature_dim = global_cond_dim
        
        # Compute max_global_tokens (may be reduced by frame selection)
        if obs_frame_selection is not None:
            rc = obs_frame_selection['recent_continuous']
            skip = obs_frame_selection['skip_every']
            n_older = len(range(n_obs_steps - rc - 1, -1, -skip))
            max_global_tokens = n_older + rc
        else:
            max_global_tokens = n_obs_steps

        print(f"Input dim: {input_dim}, Global cond dim: {global_cond_dim}")
        print(f"Max global tokens: {max_global_tokens}"
              f"{f' (selected from {n_obs_steps} frames)' if obs_frame_selection else ''}")

        model = AttentionConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,  # Not using local conditioning
            global_cond_dim=global_cond_dim,
            target_dim=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            max_global_tokens=max_global_tokens,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout
        )

        self.obs_encoder = obs_encoder
        if inference_loading == False:
            if initialize_obs_encoder is not None: 
                print(f"Loading obs encoder from {initialize_obs_encoder}")
                state_dict = torch.load(initialize_obs_encoder, map_location='cpu')
                self.obs_encoder.load_state_dict(state_dict, strict=True)
            
                if freeze_self_trained_obs_encoder: 
                    for param in self.obs_encoder.parameters(): 
                        param.requires_grad = False
        
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        # Create DDIM sampler
        DDIM_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_scheduler.num_train_timesteps,
            beta_start=self.noise_scheduler.beta_start,
            beta_end=self.noise_scheduler.beta_end,
            beta_schedule=self.noise_scheduler.beta_schedule,
            clip_sample=self.noise_scheduler.clip_sample,
            prediction_type=self.noise_scheduler.prediction_type,
        )
        DDIM_noise_scheduler.set_timesteps(num_DDIM_inference_steps)
        self.DDIM_noise_scheduler = DDIM_noise_scheduler
        self.num_DDIM_inference_steps = num_DDIM_inference_steps
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,  # Not using obs in trajectory since we use global conditioning
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        # self.project_obs_embedding = project_obs_embedding
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.one_hot_encoding_dim = one_hot_encoding_dim
        self.use_target_cond = use_target_cond
        self.max_obs_steps = n_obs_steps
        self.kwargs = kwargs

        # --- Frame selection (sparse obs conditioning) ---
        self.obs_frame_selection = obs_frame_selection
        self.max_global_tokens = max_global_tokens

        # --- Per-stream temporal attention ---
        self.use_temporal_attention = use_temporal_attention
        self.temporal_encoders = nn.ModuleDict()
        if use_temporal_attention:
            # Determine per-camera dim from encoder output
            rgb_keys = sorted([k for k, v in obs_shape_meta.items()
                               if v.get('type', 'low_dim') == 'rgb'])
            low_dim_total = sum(
                obs_shape_meta[k]['shape'][-1] for k, v in obs_shape_meta.items()
                if v.get('type', 'low_dim') == 'low_dim')
            per_camera_dim = (obs_feature_dim - low_dim_total) // len(rgb_keys)
            self._temporal_rgb_keys = rgb_keys
            self._temporal_low_dim_total = low_dim_total
            self._temporal_per_camera_dim = per_camera_dim

            for key in rgb_keys:
                self.temporal_encoders[key] = TemporalEncoder(
                    dim=per_camera_dim,
                    depth=temporal_depth,
                    num_heads=temporal_num_heads,
                    max_frames=temporal_max_frames,
                )
            t_params = sum(p.numel() for enc in self.temporal_encoders.values()
                           for p in enc.parameters())
            print(f"Per-stream temporal attention: {len(rgb_keys)} streams × "
                  f"depth={temporal_depth}, heads={temporal_num_heads}, "
                  f"dim={per_camera_dim} ({t_params:,} params)")

        # --- Prediction horizon ---
        self._unet_alignment = 4
        self.n_past_action_steps = n_past_action_steps
        self.n_future = horizon - n_obs_steps

        if n_past_action_steps is not None:
            # Static prediction horizon (when context length is fixed)
            prediction_horizon = n_past_action_steps + self.n_future
            assert prediction_horizon >= n_action_steps, \
                f"prediction_horizon ({prediction_horizon}) must be >= " \
                f"n_action_steps ({n_action_steps})"
            self.prediction_horizon = prediction_horizon
            self._action_offset = horizon - prediction_horizon
            print(f"Prediction horizon: {prediction_horizon} "
                  f"(past={n_past_action_steps}, future={self.n_future}, "
                  f"offset={self._action_offset})")
        else:
            self.prediction_horizon = horizon
            self._action_offset = 0

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        # if project_obs_embedding:
        #     print("Vision projector params: %e" % sum(p.numel() for p in self.obs_embedding_projector.parameters()))

    # ========= UNet alignment helpers ============

    def _pad_to_unet(self, x):
        """Pad sequence dim to be divisible by UNet alignment. Returns (padded, orig_len)."""
        T = x.shape[1]
        remainder = T % self._unet_alignment
        if remainder == 0:
            return x, T
        pad_len = self._unet_alignment - remainder
        x = F.pad(x, (0, 0, 0, pad_len))
        return x, T

    def _unpad_from_unet(self, x, orig_len):
        """Remove padding added by _pad_to_unet."""
        return x[:, :orig_len, :]

    def _prepare_variable_length_conditioning(self, obs_dict: Dict[str, torch.Tensor],
                                            num_obs_tokens: int) -> tuple:
        """
        Prepare variable-length observation conditioning with consistent temporal positioning.
        
        Args:
            obs_dict: Dictionary of observations with shape (B, T, ...)
            num_obs_tokens: Number of observation tokens to use (between min_obs_steps and n_obs_steps)
            
        Returns:
            global_cond: (B, num_tokens, obs_feature_dim)
            global_mask: (B, num_tokens) 
            temporal_positions: (B, num_tokens)
        """
        # Get observation features from the most recent observations
        value = next(iter(obs_dict.values()))
        B, To = value.shape[:2]
        
        # Always use the most recent observations for consistency
        start_idx = max(0, To - num_obs_tokens)
        obs_slice = slice(start_idx, To)
        
        # Extract and encode observations
        sliced_obs = dict_apply(obs_dict, lambda x: x[:, obs_slice, ...])
        # Reshape for encoder: (B, num_tokens, ...) -> (B*num_tokens, ...)
        encoder_obs = dict_apply(sliced_obs, lambda x: x.reshape(-1, *x.shape[2:]))
        
        # Get observation features
        obs_features = self.obs_encoder(encoder_obs)
        if self.project_obs_embedding:
            obs_features = self.obs_embedding_projector(obs_features)
            
        # Reshape back to (B, num_tokens, obs_feature_dim)
        obs_features = obs_features.reshape(B, num_obs_tokens, -1)
        
        # Create temporal positions - CRITICAL: use absolute positions from original sequence
        # This ensures consistent positional encoding regardless of num_obs_tokens
        temporal_positions = torch.arange(start_idx, To, device=obs_features.device)
        temporal_positions = temporal_positions.unsqueeze(0).expand(B, -1)
        
        # Create mask (all observations are valid)
        global_mask = torch.ones(B, num_obs_tokens, dtype=torch.bool, device=obs_features.device)
        
        return obs_features, global_mask, temporal_positions
    
    def _apply_per_stream_temporal(self, embeddings_full, global_mask):
        """
        Apply per-stream temporal attention to encoded features.

        Args:
            embeddings_full: (B, T, obs_feature_dim) encoded features
            global_mask: (B, T) True for valid positions
        Returns:
            (B, T, obs_feature_dim) temporally enriched features
        """
        B, T, D = embeddings_full.shape
        low_dim = self._temporal_low_dim_total
        per_cam = self._temporal_per_camera_dim
        padding_mask = ~global_mask  # True for PADDED

        # Split: [low_dim, cam1, cam2, ...]
        parts = []
        offset = 0
        # Low-dim features pass through without temporal attention
        parts.append(embeddings_full[:, :, :low_dim])
        offset = low_dim

        # Each camera stream gets temporal attention independently
        for key in self._temporal_rgb_keys:
            stream = embeddings_full[:, :, offset:offset + per_cam]  # (B, T, per_cam)
            stream = self.temporal_encoders[key](
                stream, key_padding_mask=padding_mask)
            parts.append(stream)
            offset += per_cam

        return torch.cat(parts, dim=-1)

    def _compute_frame_indices(self, T):
        """
        Compute which frames to select from a T-length buffer.
        Returns sorted index array.
        """
        rc = self.obs_frame_selection['recent_continuous']
        skip = self.obs_frame_selection['skip_every']
        recent_start = T - rc
        recent = list(range(recent_start, T))
        older = list(range(recent_start - 1, -1, -skip))[::-1]
        return older + recent

    def _prepare_batch_and_apply_obs_encoding(self, nobs, num_obs_steps_in_this):
        B = nobs[next(iter(nobs))].shape[0]
        N = nobs[next(iter(nobs))].shape[1]

        # --- Frame selection: pick subset of frames before encoding ---
        if self.obs_frame_selection is not None:
            # For each sample, select frames from [0, K_i) based on its obs count
            # All samples in a batch use the same selection pattern relative to
            # their valid frames (right-aligned)
            K = num_obs_steps_in_this  # (B,)
            max_K = K.max().item()
            selected_indices = self._compute_frame_indices(max_K)
            n_selected = len(selected_indices)
            sel_tensor = torch.tensor(selected_indices, device=self.device)

            # Select frames: (B, N, ...) -> (B, n_selected, ...)
            nobs = dict_apply(nobs, lambda x: x[:, sel_tensor])

            # Build mask: a selected index is valid if it < K_i for that sample
            sel_expanded = sel_tensor.unsqueeze(0)  # (1, n_selected)
            K_expanded = K.unsqueeze(1)  # (B, 1)
            global_mask = sel_expanded < K_expanded  # (B, n_selected)

            # Encode selected frames
            x_flat = dict_apply(nobs, lambda x: x[global_mask])
            embeddings = self.obs_encoder(x_flat)

            embeddings_full = embeddings.new_zeros((B, n_selected, self.obs_feature_dim))
            embeddings_full[global_mask] = embeddings

            # Apply per-stream temporal attention if enabled
            if self.use_temporal_attention:
                embeddings_full = self._apply_per_stream_temporal(
                    embeddings_full, global_mask)

            # Position encoding: use the original temporal indices (right-aligned)
            # so the UNet knows real temporal spacing
            position_mask = torch.where(
                global_mask,
                sel_expanded + (self.n_obs_steps - K_expanded),
                torch.zeros(1, dtype=torch.long, device=self.device))

            return embeddings_full, global_mask, position_mask

        # --- Standard path (no frame selection) ---
        K = num_obs_steps_in_this.unsqueeze(1)

        arrange_N = torch.arange(N, device=self.device)
        left_mask = arrange_N.unsqueeze(0) < K

        x_selected = dict_apply(nobs, lambda x: x[left_mask])

        embeddings = self.obs_encoder(x_selected)

        embeddings_full = embeddings.new_zeros((B, N, self.obs_feature_dim))
        embeddings_full[left_mask] = embeddings
        embeddings_full = embeddings_full[:, :self.n_obs_steps, :]

        global_mask = left_mask[:, :self.n_obs_steps]

        # Apply per-stream temporal attention if enabled
        if self.use_temporal_attention:
            embeddings_full = self._apply_per_stream_temporal(
                embeddings_full, global_mask)

        idx = torch.arange(self.n_obs_steps, device=self.device).unsqueeze(0)
        first_k = idx < K
        position_mask = torch.where(first_k, idx + (self.n_obs_steps - K), torch.zeros_like(idx))

        return embeddings_full, global_mask, position_mask

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None, global_mask=None,
            target_cond=None, temporal_positions=None, generator=None,
            use_DDIM=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
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
            # 1. apply conditioning
            trajectory[condition_mask_pad] = condition_data_pad[condition_mask_pad]

            # 2. predict model output with attention-based conditioning
            model_output = model(trajectory, t,
                global_cond=global_cond, global_mask=global_mask,
                target_cond=target_cond, local_cond=local_cond,
                temporal_positions=temporal_positions)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask_pad] = condition_data_pad[condition_mask_pad]
        trajectory = self._unpad_from_unet(trajectory, orig_len)
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], 
                      num_obs_tokens: Optional[int] = None,
                      use_DDIM=False, use_custom_inference_tokens=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs"
        - if use_target_cond is true, obs_dict must also include "target"
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        if self.use_target_cond:
            assert 'target' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        
        # normalize input
        nobs = self.normalizer.normalize(obs_dict['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(obs_dict['target'])
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        if use_custom_inference_tokens is not None:
            To = min(To, use_custom_inference_tokens)
        Da = self.action_dim

        # Determine number of observation tokens to use
        if num_obs_tokens is None:
            if use_custom_inference_tokens is not None:
                def shifting_tokens(tensor):
                    new_tensor = tensor.new_zeros(tensor.shape)
                    new_tensor[:, :use_custom_inference_tokens, ...] = tensor[:, -use_custom_inference_tokens:, ...]
                    return new_tensor

                nobs = dict_apply(nobs, shifting_tokens)
                num_obs_tokens = torch.ones(B, dtype=torch.long, device=value.device) * use_custom_inference_tokens
            else:
                num_obs_tokens = torch.ones(B, dtype=torch.long, device=value.device) * self.max_obs_steps

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

        # build input
        device = self.device
        dtype = self.dtype

        # Prepare variable-length conditioning
        global_cond, global_mask, temporal_positions = self._prepare_batch_and_apply_obs_encoding(nobs, num_obs_tokens)

        # Append one hot encoding if needed
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = obs_dict['one_hot_encoding']
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(-1, num_obs_tokens, -1)
            global_cond = torch.cat([global_cond, one_hot_expanded], dim=-1)

        # Empty data for action (no conditioning on actions during inference)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(B, -1)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            global_mask=global_mask,
            target_cond=target_cond,
            temporal_positions=temporal_positions,
            use_DDIM=use_DDIM,
            **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[...,:Da]
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

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch, noisy_trajectory, timesteps):
        """
        Forward pass with variable-length observation conditioning.
        """
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]

        # Check if batch has pre-computed observation lengths (from dataset)
            # Use dataset-level variable observation lengths
        
        global_cond, global_mask, temporal_positions = self._prepare_batch_and_apply_obs_encoding(nobs, batch['sample_metadata']['num_obs_steps'])

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch['one_hot_encoding']
            # Expand to match observation tokens
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(-1, num_obs_tokens, -1)
            global_cond = torch.cat([global_cond, one_hot_expanded], dim=-1)
        
        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1) # B, D_t

        # Pad for UNet alignment if needed
        noisy_trajectory, orig_len = self._pad_to_unet(noisy_trajectory)
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=None, global_cond=global_cond, global_mask=global_mask,
            target_cond=target_cond, temporal_positions=temporal_positions)
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

        To = num_obs_steps  # (B,)
        past_in_pred = torch.clamp(To, max=self.n_past_action_steps)
        pred_lens = past_in_pred + self.n_future
        offsets = To - past_in_pred

        max_pred = pred_lens.max().item()
        padded_pred = math.ceil(max_pred / self._unet_alignment) * self._unet_alignment

        trajectory = nactions.new_zeros((B, padded_pred, Da))
        loss_mask = torch.zeros((B, padded_pred, Da), dtype=torch.bool, device=device)

        for i in range(B):
            pl = pred_lens[i].item()
            off = offsets[i].item()
            trajectory[i, :pl, :] = nactions[i, off:off + pl, :]
            loss_mask[i, :pl, :] = True

        return trajectory, loss_mask

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        batch_size = nactions.shape[0]

        num_obs_steps = batch['sample_metadata']['num_obs_steps']
        global_cond, global_mask, temporal_positions = self._prepare_batch_and_apply_obs_encoding(nobs, num_obs_steps)

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch['one_hot_encoding']
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(-1, global_cond.shape[1], -1)
            global_cond = torch.cat([global_cond, one_hot_expanded], dim=-1)

        # handle target conditioning
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

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask
        if dynamic_loss_mask is not None:
            loss_mask = loss_mask & dynamic_loss_mask

        # Pad for UNet alignment if needed
        noisy_trajectory, orig_len = self._pad_to_unet(noisy_trajectory)
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=None, global_cond=global_cond, global_mask=global_mask,
            target_cond=target_cond, temporal_positions=temporal_positions)
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
        
        # Update training step for curriculum learning        
        return loss

    def compute_obs_embedding(self, batch, num_obs_tokens: Optional[int] = None):
        """
        Compute observation embeddings for analysis.
        """
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        
        # Check if batch has pre-computed observation lengths (from dataset)
        if 'obs_lengths' in batch:
            # Use dataset-level variable observation lengths
            global_cond, global_mask, temporal_positions = self._prepare_variable_length_conditioning_batch(
                nobs, batch['obs_lengths'])
        else:
            # Fallback: use specified or default observation tokens
            if num_obs_tokens is None:
                num_obs_tokens = self.max_obs_steps
            global_cond, global_mask, temporal_positions = self._prepare_variable_length_conditioning(
                nobs, num_obs_tokens)
        
        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch['one_hot_encoding']
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(-1, num_obs_tokens, -1)
            global_cond = torch.cat([global_cond, one_hot_expanded], dim=-1)

        return global_cond, global_mask, temporal_positions