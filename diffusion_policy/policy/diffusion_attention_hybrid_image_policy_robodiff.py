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
# import robomimic.models.obs_core as rmobsc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


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
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
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
        
        print(f"Input dim: {input_dim}, Global cond dim: {global_cond_dim}")
        print(f"Max global tokens: {n_obs_steps}")

        model = AttentionConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,  # Not using local conditioning
            global_cond_dim=global_cond_dim,
            target_dim=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            max_global_tokens=n_obs_steps,
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
        self.max_global_tokens = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        # if project_obs_embedding:
        #     print("Vision projector params: %e" % sum(p.numel() for p in self.obs_embedding_projector.parameters()))

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
    
    def _prepare_batch_and_apply_obs_encoding(self, nobs, num_obs_steps_in_this): 
        B = nobs[next(iter(nobs))].shape[0]
        N = nobs[next(iter(nobs))].shape[1]

        K = num_obs_steps_in_this.unsqueeze(1)

        arrange_N = torch.arange(N, device=self.device)
        left_mask = arrange_N.unsqueeze(0) < K 

        x_selected = dict_apply(nobs, lambda x: x[left_mask])

        embeddings = self.obs_encoder(x_selected)

        embeddings_full = embeddings.new_zeros((B, N, self.obs_feature_dim))
        embeddings_full[left_mask] = embeddings
        embeddings_full = embeddings_full[:, :self.n_obs_steps, :]

        global_mask = left_mask[:, :self.n_obs_steps]

        idx = torch.arange(self.n_obs_steps, device=self.device).unsqueeze(0)           # [1, N] -> broadcasts to [B, N]
        first_k = idx < K                                             # [B, N], True for positions 0..K_b-1
        position_mask = torch.where(first_k, idx + (self.n_obs_steps - K), torch.zeros_like(idx))  # [B, N], int64

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

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

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
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], 
                      num_obs_tokens: Optional[int] = None,
                      use_DDIM=False) -> Dict[str, torch.Tensor]:
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
        T = self.horizon
        Da = self.action_dim
        
        # Determine number of observation tokens to use
        if num_obs_tokens is None:
            # assume it is not training but rather sampling 
            num_obs_tokens = torch.ones(B, dtype=torch.long, device=value.device) * self.max_obs_steps
        
        # num_obs_tokens = min(num_obs_tokens, To)  # Can't use more tokens than available
        
        # build input
        device = self.device
        dtype = self.dtype

        # Prepare variable-length conditioning
        global_cond, global_mask, temporal_positions = self._prepare_batch_and_apply_obs_encoding(nobs, num_obs_tokens)
        
        # Append one hot encoding if needed
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = obs_dict['one_hot_encoding']
            # Expand one-hot to match number of observation tokens
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(-1, num_obs_tokens, -1)
            global_cond = torch.cat([global_cond, one_hot_expanded], dim=-1)

        # Empty data for action (no conditioning on actions during inference)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(B, -1) # B, D_t

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

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred,
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

        # Predict the noise residual
        return self.model(noisy_trajectory, timesteps, 
            local_cond=None, global_cond=global_cond, global_mask=global_mask,
            target_cond=target_cond, temporal_positions=temporal_positions)

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Check if batch has pre-computed observation lengths (from dataset)
        global_cond, global_mask, temporal_positions = self._prepare_batch_and_apply_obs_encoding(nobs, batch['sample_metadata']['num_obs_steps'])

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            one_hot_encoding = batch['one_hot_encoding']
            one_hot_expanded = one_hot_encoding.unsqueeze(1).expand(-1, num_obs_tokens, -1)
            global_cond = torch.cat([global_cond, one_hot_expanded], dim=-1)
        
        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1) # B, D_t

        # trajectory for training is just the actions (no obs concatenation)
        trajectory = nactions

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the actions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning (not used since we don't condition on actions)
        # noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=None, global_cond=global_cond, global_mask=global_mask,
            target_cond=target_cond, temporal_positions=temporal_positions)

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