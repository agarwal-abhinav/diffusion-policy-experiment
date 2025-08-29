from typing import Dict
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
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmobsc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

from typing import List, Optional

class ImageRandomCrop(nn.Module): 
    def __init__(self, crop_size: tuple[int, int]): 
        super().__init__()
        self.ch, self.cw = crop_size

    @torch.no_grad()
    def _sample_positions(self, B, H, W, device, training: bool): 
        if training: 
            top = torch.randint(0, H - self.ch + 1, (B,), device=device)
            left = torch.randint(0, W - self.cw + 1, (B,), device=device)
        else:
            top = torch.full((B,), (H - self.ch)//2, device=device, dtype=torch.long)
            left = torch.full((B,), (W - self.cw)//2, device=device, dtype=torch.long)
        return top, left

    def forward(self, x): 
        """
        x: B, T, C, H, W -> B, T*C, H, W then crop -> B, T*C, crop_H, crop_W
        """
        B, T, C, H, W = x.shape 
        
        # Reshape to B, T*C, H, W
        x = x.reshape(B, T*C, H, W)

        assert x.dtype == torch.float32

        top, left = self._sample_positions(B, H, W, x.device, self.training)

        h_idx = top[:, None] + torch.arange(self.ch, device=x.device)[None, :]
        w_idx = left[:, None] + torch.arange(self.cw, device=x.device)[None, :]

        idx_h = h_idx[:, None, :, None].expand(B, T*C, self.ch, W)
        x_h = x.gather(dim=2, index=idx_h)

        idx_w = w_idx[:, None, None, :].expand(B, T*C, self.ch, self.cw)
        x_hw = x_h.gather(dim=3, index=idx_w)

        return x_hw

class ResNet18ObsEncoder(nn.Module): 
    def __init__(self, 
                 output_dim: int,
                 n_obs_steps: int,  # T parameter for T*C channels
                 pretrained: Optional[str] = None, 
                 use_pretrained_imagenet: bool = False,
                 normalize_output: bool = True, 
                 random_crop_size: Optional[tuple[int, int]] = (112, 112)): 
        super().__init__()

        if random_crop_size is not None: 
            self.random_cropper = ImageRandomCrop(random_crop_size)
        else: 
            self.random_cropper = None

        self.n_obs_steps = n_obs_steps
        
        # Create ResNet18 but modify it for 112x112 input
        if use_pretrained_imagenet:
            self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet18(weights=None)
        
        # Modify the first convolutional layer to accept T*C channels
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=n_obs_steps * 3,  # T * C where C=3 for RGB
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )
        
        # For 112x112 input, we need to modify the architecture to prevent too much downsampling
        # Standard ResNet18: 224->112 (conv1) -> 56 (maxpool) -> 28->14->7 (blocks)
        # For 112x112: 112->56 (conv1) -> 28 (maxpool) -> 14->7->4 (blocks) - too small!
        # Solution: Reduce stride in first conv or remove/modify maxpool
        
        # Option 1: Keep stride=2 in conv1 but reduce maxpool stride
        original_maxpool = self.backbone.maxpool
        self.backbone.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # If using pretrained weights, we need to adapt the first layer weights
        if use_pretrained_imagenet:
            # Average the original 3-channel weights across T repetitions
            original_weight = original_conv1.weight  # [64, 3, 7, 7]
            # Repeat the weights T times along the channel dimension
            new_weight = original_weight.repeat(1, n_obs_steps, 1, 1)  # [64, T*3, 7, 7]
            # Scale by 1/T to maintain similar activation magnitudes
            new_weight = new_weight / n_obs_steps
            self.backbone.conv1.weight.data = new_weight
            
            if original_conv1.bias is not None:
                self.backbone.conv1.bias.data = original_conv1.bias.data.clone()

        # Load custom pretrained weights if provided
        if pretrained is not None: 
            sd = torch.load(pretrained, map_location='cpu')
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            if missing:
                print(f"Missing keys when loading pretrained ResNet18: {missing}")
            if unexpected:
                print(f"Unexpected keys when loading pretrained ResNet18: {unexpected}")

        # Remove the final classification layer 
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.proj = nn.Linear(feat_dim, output_dim)

        self.normalize_output = normalize_output
    
    def forward(self, x): 
        """
        x: B, T, C, H, W
        """
        if self.random_cropper is not None:
            x = self.random_cropper(x)  # B, T*C, crop_H, crop_W
        else:
            # Reshape from B, T, C, H, W to B, T*C, H, W
            B, T, C, H, W = x.shape
            x = x.reshape(B, T*C, H, W)
                         
        x = self.backbone(x)
        x = self.proj(x)

        if self.normalize_output:
            x = F.normalize(x, dim=-1)

        return x
    
class RGBInputResNet18Encoder(nn.Module): 
    def __init__(self, 
                 input_labels: List[str], 
                 output_dim: int,
                 n_obs_steps: int,
                 pretrained: Optional[str] = None, 
                 use_pretrained_imagenet: bool = False,
                 normalize_output: bool = True, 
                 random_crop_size: Optional[tuple[int, int]] = (112, 112)
                 ): 
        super().__init__() 

        self.module_dict = nn.ModuleDict()

        for key in input_labels: 
            self.module_dict[key] = ResNet18ObsEncoder(
                output_dim=output_dim,
                n_obs_steps=n_obs_steps,
                pretrained=pretrained, 
                use_pretrained_imagenet=use_pretrained_imagenet,
                normalize_output=normalize_output, 
                random_crop_size=random_crop_size
            )
        
        self.keys = input_labels
    
    def forward(self, x_dict): 
        outputs = []
        for key in self.keys: 
            outputs.append(self.module_dict[key](x_dict[key]))

        return torch.cat(outputs, dim=-1)

class AllInputEncoder(nn.Module): 
    def __init__(self, 
                 rgb_input_labels: List[str],
                 other_input_labels: List[str], 
                 video_output_dim_per_modality: int,
                 n_obs_steps: int,
                 pretrained_rgb_encoder: Optional[str] = None, 
                 use_pretrained_imagenet: bool = False,
                 normalize_rgb_output: bool = True, 
                 crop_shape: Optional[tuple[int, int]] = (112, 112)
                 ): 
        super().__init__()

        self.rgb_encoder = RGBInputResNet18Encoder(
            input_labels=rgb_input_labels,
            output_dim=video_output_dim_per_modality,
            n_obs_steps=n_obs_steps,
            pretrained=pretrained_rgb_encoder,
            use_pretrained_imagenet=use_pretrained_imagenet,
            normalize_output=normalize_rgb_output, 
            random_crop_size=crop_shape
        )
        self.other_input_labels = other_input_labels
        self.rgb_input_labels = rgb_input_labels    
    
    def forward(self, x_dict): 
        rgb_features = self.rgb_encoder(x_dict)

        all_features = [rgb_features]

        for key in self.other_input_labels: 
            all_features.append(x_dict[key].view(x_dict[key].shape[0], -1))

        return torch.cat(all_features, dim=-1)
    
class ImageNormalizer(nn.Module): 
    def __init__(self, mean, std, inplace=False): 
        super().__init__()

        mean = torch.as_tensor(mean).view(1, 1, -1, 1, 1)
        std = torch.as_tensor(std).view(1, 1, -1, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.inplace = inplace

    def forward(self, x): 
        if self.inplace: 
            x.sub_(self.mean).div_(self.std)
            return x 
        else: 
            return (x - self.mean) / self.std

class DiffusionUnetHybridImageTargetedPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            video_output_dim_per_modality,
            num_inference_steps=None,
            obs_as_global_cond=True,
            use_target_cond=False,
            target_dim=None,
            crop_shape=(224, 224),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_embedding_dim=None,
            one_hot_encoding_dim=0,
            obs_encoder_group_norm=False,
            num_DDIM_inference_steps=10,
            pretrained_encoder: Optional[str] = None,
            use_pretrained_imagenet: bool = False,
            normalize_rgb_output: bool = True,
            freeze_encoder: bool = False, 
            # parameters passed to step
            **kwargs):
        super().__init__()

        if use_target_cond:
            assert target_dim is not None
        assert one_hot_encoding_dim >= 0
        assert obs_as_global_cond

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
                raise RuntimeError(f"Unsupported obs type: {type}")

        obs_encoder = AllInputEncoder(
            rgb_input_labels=obs_config['rgb'], 
            other_input_labels=obs_config['low_dim'],
            video_output_dim_per_modality=video_output_dim_per_modality,
            n_obs_steps=n_obs_steps,
            pretrained_rgb_encoder=pretrained_encoder,
            use_pretrained_imagenet=use_pretrained_imagenet,
            normalize_rgb_output=normalize_rgb_output, 
            crop_shape=crop_shape
        )

        assert obs_embedding_dim is None
        if obs_embedding_dim is not None:
            obs_feature_dim = obs_embedding_dim
            self.obs_embedding_projector = MLP(obs_encoder.output_shape()[0], [], obs_feature_dim)
            self.obs_embedding_projector.to('cuda' if torch.cuda.is_available() else 'cpu')
            project_obs_embedding = True
        else:
            obs_feature_dim = len(obs_config['rgb']) * video_output_dim_per_modality + sum(
                [math.prod(obs_key_shapes[key]) * n_obs_steps for key in obs_config['low_dim']])
            project_obs_embedding = False

        # create diffusion model
        # obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim 
        print(f"Input dim: {input_dim}, Global cond dim: {global_cond_dim}")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            target_dim=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        
        if freeze_encoder: 
            for param in self.obs_encoder.parameters(): 
                param.requires_grad = False
        
        self.model = model
        self.noise_scheduler = noise_scheduler

        # ImageNet normalization values
        mean = (0.485, 0.456, 0.406)
        std  = (0.229, 0.224, 0.225)

        self.image_normalizer = nn.ModuleDict()
        for key in obs_config['rgb']:
            self.image_normalizer[key] = ImageNormalizer(mean, std, inplace=True)

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
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.project_obs_embedding = project_obs_embedding
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.one_hot_encoding_dim = one_hot_encoding_dim
        self.use_target_cond = use_target_cond
        self.kwargs = kwargs
        self.obs_config = obs_config

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        if project_obs_embedding:
            print("Vision projector params: %e" % sum(p.numel() for p in self.obs_embedding_projector.parameters()))
        print("Global cond dim: %d" % (0 if global_cond_dim is None else global_cond_dim))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            target_cond=None, generator=None,
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

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond,
                target_cond=target_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory
    
    def _apply_obs_normalization(self, obs_dict): 
        nobs = dict()
        for key in self.obs_config['rgb']:
            nobs[key] = self.image_normalizer[key](obs_dict[key]).to(torch.float32)
        
        for key in self.obs_config['low_dim']:
            nobs[key] = self.normalizer[key].normalize(obs_dict[key]).to(torch.float32)

        return nobs

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], use_DDIM=False) -> Dict[str, torch.Tensor]:
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
        nobs = self._apply_obs_normalization(obs_dict['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(obs_dict['target'])
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...])
            global_cond = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            # global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = obs_dict['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)

        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(B, -1) # B, D_t

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            target_cond=target_cond,
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
            'action_pred': action_pred
        }
        return result

    def compute_obs_embedding(self, batch):
        assert 'valid_mask' not in batch
        nobs = self._apply_obs_normalization(batch['obs'])
        key = next(iter(nobs.keys()))
        batch_size = nobs[key].shape[0]

        # handle different ways of passing observation
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...])
            global_cond = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            # global_cond = nobs_features.reshape(batch_size, -1)
        else:
            nactions = self.normalizer['action'].normalize(batch['action'])
            horizon = nactions.shape[1]
            trajectory = nactions
            cond_data = trajectory
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        
        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = batch['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)

        return global_cond

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_encoder_output(self, batch): 
        assert 'valid_mask' not in batch
        nobs = self._apply_obs_normalization(batch['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # print(f"Inside batch size: {batch_size}")

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...])
            global_cond = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            # global_cond = nobs_features.reshape(batch_size, -1)

        return global_cond, nobs_features

    def forward(self, batch, noisy_trajectory, timesteps):
        assert 'valid_mask' not in batch
        nobs = self._apply_obs_normalization(batch['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # print(f"Inside batch size: {batch_size}")

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...])
            global_cond = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            # global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = batch['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)
        
        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1) # B, D_t

        # Predict the noise residual
        return self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond,
            target_cond=target_cond)

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self._apply_obs_normalization(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...])
            global_cond = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            # global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        
        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = batch['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)
        
        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1) # B, D_t

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond,
            target_cond=target_cond)

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