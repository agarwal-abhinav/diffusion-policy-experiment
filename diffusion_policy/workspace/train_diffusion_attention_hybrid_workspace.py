if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf, ListConfig
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import dill
import shutil
import torch
from torch.nn.parallel import DataParallel
from torch.cuda.amp import GradScaler, autocast
from einops import rearrange, reduce
import torch.nn.functional as F
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_attention_hybrid_image_policy import DiffusionAttentionHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class TrainDiffusionAttentionHybridWorkspace(BaseWorkspace):
    """
    Workspace for training attention-based diffusion policies.
    Supports:
    - Variable-length observation sequences
    - Attention-based conditioning with temporal positional encoding
    - Curriculum learning for observation length
    - Enhanced logging for attention analysis
    """
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        # This environment does not support impainting or local conditioning
        assert cfg.policy.obs_as_global_cond is True

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        num_GPU = torch.cuda.device_count()
        self.model = hydra.utils.instantiate(cfg.policy)
        if num_GPU > 0:
            self.model = self.model.to(torch.device("cuda"))
        else:
            self.model = self.model.to(torch.device("cpu"))

        print(f"Running attention model on {num_GPU} GPU(s).")
        self.model = DataParallelWrapper(self.model, device_ids=range(num_GPU))

        # Load pretrained model if finetuning
        if 'pretrained_checkpoint' in cfg and cfg.pretrained_checkpoint is not None:
            print(f"Loading pretrained model from {cfg.pretrained_checkpoint}.")
            path = pathlib.Path(cfg.pretrained_checkpoint)
            payload = torch.load(path.open('rb'), pickle_module=dill)
            self.model.load_state_dict(payload['state_dicts']['model'])
        else:
            print("Initializing attention model using default parameters.")

        self.ema_model: DiffusionAttentionHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state with variable learning rates
        self.encoder_lr_scale = getattr(cfg.training, 'encoder_lr_scale', 1.0)
        self.use_variable_lr = getattr(cfg.training, 'use_variable_lr', False)
        
        if self.use_variable_lr and hasattr(self.model, 'obs_encoder'):
            print(f"Using variable learning rate: encoder LR = {cfg.optimizer.lr * self.encoder_lr_scale:.6f}, rest = {cfg.optimizer.lr:.6f}")
            
            # Separate parameters for encoder and rest of model
            encoder_params = list(self.model.obs_encoder.parameters())
            encoder_param_ids = {id(p) for p in encoder_params}
            
            other_params = [p for p in self.model.parameters() if id(p) not in encoder_param_ids]
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': encoder_params, 'lr': cfg.optimizer.lr * self.encoder_lr_scale},
                {'params': other_params, 'lr': cfg.optimizer.lr}
            ]
            
            # Create optimizer with parameter groups
            import torch.optim as optim
            optimizer_class = getattr(optim, cfg.optimizer._target_.split('.')[-1])  # Extract class name
            
            optimizer_kwargs = dict(cfg.optimizer)
            optimizer_kwargs.pop('_target_')
            optimizer_kwargs.pop('lr')  # Remove base lr since we're setting per group
            
            self.optimizer = optimizer_class(param_groups, **optimizer_kwargs)
        else:
            self.optimizer = hydra.utils.instantiate(
                cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

        self.new_type_dataloader = getattr(cfg, 'new_type_dataloader', False)
        
        # configure mixed precision training
        self.use_amp = getattr(cfg.training, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        print(f"Mixed precision training: {'enabled' if self.use_amp else 'disabled'}")

        # NEW: Attention-specific settings
        self.log_attention_stats = getattr(cfg.training, 'log_attention_stats', True)
        self.attention_curriculum_steps = getattr(cfg.training, 'attention_curriculum_steps', 10000)
        print(f"Attention curriculum steps: {self.attention_curriculum_steps}")
        print(f"Attention stats logging: {'enabled' if self.log_attention_stats else 'disabled'}")

    def _process_variable_length_batch(self, batch, rgb_keys):
        """
        Process a batch with variable observation lengths per sample.
        
        Args:
            batch: Batch dict with 'obs', 'action', 'sample_metadata'
            rgb_keys: List of RGB observation keys
        
        Returns:
            Processed batch with proper formatting and encoding
        """
        batch_size = batch['action'].shape[0]
        
        # Extract observation lengths for each sample in the batch
        # DataLoader collates sample_metadata into dict of tensors/lists
        obs_lengths = []
        if 'sample_metadata' in batch:
            metadata = batch['sample_metadata'] 
            if 'num_obs_steps' in metadata:
                # Metadata is already collated as tensors
                if isinstance(metadata['num_obs_steps'], torch.Tensor):
                    obs_lengths = metadata['num_obs_steps'].tolist()
                else:
                    obs_lengths = list(metadata['num_obs_steps'])
            else:
                # Fallback: use max_obs_steps or default
                default_steps = metadata.get('max_obs_steps', [8] * batch_size)
                if isinstance(default_steps, torch.Tensor):
                    obs_lengths = default_steps.tolist()
                else:
                    obs_lengths = list(default_steps)
        else:
            # No metadata available, use default
            obs_lengths = [8] * batch_size
        
        # Process each observation type
        for key in rgb_keys:
            if key in batch['obs']:
                # Process RGB images: moveaxis and normalize, handle NaN padding
                obs_tensor = batch['obs'][key]  # (B, T, H, W, 3) - already float32 with NaN padding
                
                # Convert to (B, T, 3, H, W) 
                obs_tensor = torch.moveaxis(obs_tensor, -1, 2)  # (B, T, 3, H, W)
                
                # Create mask for valid observations (non-NaN regions)
                valid_mask = ~torch.isnan(obs_tensor[:, :, 0, 0, 0])  # Check first pixel for NaN
                
                # Normalize valid regions only (divide by 255.0)
                # NaN regions will remain NaN after division
                obs_tensor = obs_tensor / 255.0
                
                # Set padded regions to 0.0 for cleaner handling
                obs_tensor[torch.isnan(obs_tensor)] = 0.0
                
                batch['obs'][key] = obs_tensor
        
        # Store observation lengths in batch for policy use
        batch['obs_lengths'] = torch.tensor(obs_lengths, dtype=torch.long, device=batch['action'].device)
        
        return batch

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # sync training step with model for curriculum learning
                if hasattr(self.model, 'training_step'):
                    self.model.training_step = self.global_step
                # self.epoch is loaded with the last completed epoch
                # the current epoch is the next epoch (hence += 1)
                self.epoch += 1

        # configure dataset and save normalizer
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, os.path.join(self.output_dir, 'normalizer.pt'))

        # configure validation datasets
        self.num_datasets = dataset.get_num_datasets()
        self.sample_probabilities = dataset.get_sample_probabilities()
        val_dataloaders = []
        for i in range(self.num_datasets):
            val_dataset = dataset.get_validation_dataset(i)
            val_dataloaders.append(DataLoader(val_dataset, **cfg.val_dataloader))
        self._print_dataset_diagnostics(cfg, dataset, train_dataloader, val_dataloaders)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        training_steps = getattr(cfg.training, 'total_train_steps', None)
        if training_steps is not None: 
            assert cfg.training.gradient_accumulate_every == 1, "Gradient accumulation not supported with total_train_steps"
            single_epoch_steps = len(train_dataloader)
            num_epochs = int(training_steps // single_epoch_steps)
            if training_steps % single_epoch_steps != 0:
                num_epochs += 1
            print(f"Training for {num_epochs} epochs to achieve {training_steps} steps.")
            cfg.training.num_epochs = num_epochs

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        assert cfg.training.checkpoint_every % cfg.training.val_every == 0
        if not isinstance(cfg.checkpoint, ListConfig):
            # configure single checkpoint manager
            topk_managers = [TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )]
            save_last_ckpt = cfg.checkpoint.save_last_ckpt
            save_last_snapshot = cfg.checkpoint.save_last_snapshot
        else:
            # configure multiple checkpoint managers
            topk_managers = []
            save_last_ckpt = False
            save_last_snapshot = False
            for ckpt_cfg in cfg.checkpoint:
                topk_managers.append(TopKCheckpointManager(
                    save_dir=os.path.join(self.output_dir, 'checkpoints'),
                    **ckpt_cfg.topk
                ))
                save_last_ckpt = save_last_ckpt or ckpt_cfg.save_last_ckpt
                save_last_snapshot = save_last_snapshot or ckpt_cfg.save_last_snapshot

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        val_sampling_batches = [None] * self.num_datasets

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # NEW: Initialize attention statistics tracking
        attention_stats = {
            'obs_lengths_used': [],
            'curriculum_progress': 0.0
        }

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        resume_epoch = self.epoch
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(resume_epoch, cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                obs_lengths_epoch = []  # Track observation lengths used this epoch
                
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        breakpoint()
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        breakpoint()
                        
                        # NEW: Update dataset training step for progressive curriculum
                        if hasattr(dataset, 'set_training_step'):
                            dataset.set_training_step(self.global_step)
                        
                        if self.new_type_dataloader: 
                            # Process variable-length observations
                            batch = self._process_variable_length_batch(batch, dataset.rgb_keys)
                        batch_size = batch['action'].shape[0]
                        
                        # NEW: Update model training step for curriculum learning
                        if hasattr(self.model, 'training_step'):
                            self.model.training_step = self.global_step
                        
                        # construct noisy trajectory
                        trajectory = self.model.normalizer['action'].normalize(batch['action'])
                        noise = torch.randn(trajectory.shape, device=trajectory.device)
                        # Sample a random timestep for each sample
                        timesteps = torch.randint(
                            0, self.model.noise_scheduler.config.num_train_timesteps, 
                            (batch_size,), device=trajectory.device
                        ).long()
                        # Add noise to the clean images according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_trajectory = self.model.noise_scheduler.add_noise(
                            trajectory, noise, timesteps)
                        
                        # Forward pass with optional mixed precision
                        if self.use_amp:
                            with autocast(dtype=torch.float16):
                                pred = self.model(batch, noisy_trajectory, timesteps)
                                raw_loss = self.compute_loss(trajectory, noise, pred)
                                loss = raw_loss / cfg.training.gradient_accumulate_every
                        else:
                            pred = self.model(batch, noisy_trajectory, timesteps)
                            raw_loss = self.compute_loss(trajectory, noise, pred)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                        
                        # NEW: Track observation lengths used (if available)
                        if hasattr(self.model, '_get_current_obs_steps'):
                            current_obs_length = self.model._get_current_obs_steps(self.global_step)
                            obs_lengths_epoch.append(current_obs_length)
                        
                        # Backward pass with optional mixed precision
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            if self.use_amp:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.optimizer.zero_grad()
                            else:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        # NEW: Add attention-specific logging
                        if self.log_attention_stats and obs_lengths_epoch:
                            step_log['attention/current_obs_length'] = obs_lengths_epoch[-1]
                            step_log['attention/curriculum_progress'] = min(1.0, self.global_step / self.attention_curriculum_steps)

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                    # End of batch
                # End of epoch

                # NEW: Update attention statistics
                if obs_lengths_epoch:
                    attention_stats['obs_lengths_used'].extend(obs_lengths_epoch)
                    attention_stats['curriculum_progress'] = min(1.0, self.global_step / self.attention_curriculum_steps)
                        
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # NEW: Add epoch-level attention statistics
                if self.log_attention_stats and obs_lengths_epoch:
                    step_log['attention/epoch_avg_obs_length'] = np.mean(obs_lengths_epoch)
                    step_log['attention/epoch_min_obs_length'] = np.min(obs_lengths_epoch)
                    step_log['attention/epoch_max_obs_length'] = np.max(obs_lengths_epoch)
                    step_log['attention/obs_length_std'] = np.std(obs_lengths_epoch)

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_loss_per_dataset = []
                        for dataset_idx in range(self.num_datasets):
                            val_dataloader = val_dataloaders[dataset_idx]
                            val_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Dataset {dataset_idx} validation, epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    if self.new_type_dataloader: 
                                        # Process variable-length observations for validation
                                        batch = self._process_variable_length_batch(batch, dataset.rgb_keys)
                                    if val_sampling_batches[dataset_idx] is None:
                                        val_sampling_batches[dataset_idx] = batch
                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            # End of validation loss computation loop

                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                val_loss_per_dataset.append(val_loss)
                                # log epoch average validation loss
                                step_log[f'val_loss_{dataset_idx}'] = val_loss
                        # End val_dataloader loop

                        # Compute overall_val_loss
                        overall_val_loss = 0
                        for i in range(self.num_datasets):
                            overall_val_loss += self.sample_probabilities[i] * val_loss_per_dataset[i]
                        step_log['val_loss'] = overall_val_loss

                # run diffusion sampling on a _single_ validation batch from each dataset
                if (self.epoch % cfg.training.sample_every) == 0 and cfg.training.log_val_mse:
                    with torch.no_grad():
                        val_ddpm_action_mses = []
                        val_ddim_action_mses = []
                        for dataset_idx in range(self.num_datasets):
                            # Get the validation batch for this dataset
                            val_sampling_batch = val_sampling_batches[dataset_idx]
                            val_batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            val_obs_dict = {key: val_batch[key] for key in val_batch.keys() if key != 'action'}
                            val_gt_action = val_sampling_batch['action']
                            
                            # NEW: Test with different observation lengths during validation
                            obs_lengths_to_test = [
                                policy.min_obs_steps,
                                policy.min_obs_steps + (policy.max_obs_steps - policy.min_obs_steps) // 2,
                                policy.max_obs_steps
                            ]
                            
                            for obs_len in obs_lengths_to_test:
                                if obs_len <= policy.max_obs_steps:
                                    # Evaluate MSE when diffusing with DDPM
                                    if cfg.training.eval_mse_DDPM:
                                        result = policy.predict_action(val_obs_dict, 
                                                                     num_obs_tokens=obs_len, 
                                                                     use_DDIM=False)
                                        pred_action = result['action_pred']
                                        mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                                        step_log[f'val_ddpm_mse_{dataset_idx}_obs{obs_len}'] = mse.item()
                                        
                                        # Only use max_obs for overall metrics
                                        if obs_len == policy.max_obs_steps:
                                            val_ddpm_action_mses.append(mse.item())
                                    
                                    # Evaluate MSE when diffusing with DDIM
                                    if cfg.training.eval_mse_DDIM:
                                        result = policy.predict_action(val_obs_dict, 
                                                                     num_obs_tokens=obs_len,
                                                                     use_DDIM=True)
                                        pred_action = result['action_pred']
                                        mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                                        step_log[f'val_ddim_mse_{dataset_idx}_obs{obs_len}'] = mse.item()
                                        
                                        # Only use max_obs for overall metrics
                                        if obs_len == policy.max_obs_steps:
                                            val_ddim_action_mses.append(mse.item())
                        
                        # Compute weighted val action MSEs (using max_obs results)
                        if cfg.training.eval_mse_DDPM and val_ddpm_action_mses:
                            val_ = 0
                            for i in range(self.num_datasets):
                                val_ += self.sample_probabilities[i] * val_ddpm_action_mses[i]
                            step_log['val_ddpm_mse'] = val_
                        if cfg.training.eval_mse_DDIM and val_ddim_action_mses:
                            val_ = 0
                            for i in range(self.num_datasets):
                                val_ += self.sample_probabilities[i] * val_ddim_action_mses[i]
                            step_log['val_ddim_mse'] = val_

                        # Clean up
                        try:
                            del val_batch, val_obs_dict, val_gt_action, result, pred_action, mse
                        except:
                            pass
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if save_last_ckpt:
                        self.save_checkpoint()
                    if save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    topk_ckpt_paths = []
                    for i, topk_manager in enumerate(topk_managers):
                        protected_ckpts = self._get_protected_paths(i, topk_managers)
                        ckpt_path = topk_manager.get_ckpt_path(metric_dict, protected_ckpts)
                        topk_ckpt_paths.append(ckpt_path)

                    for i, topk_ckpt_path in enumerate(topk_ckpt_paths):
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                            break
                # last epoch => save last checkpoint
                if self.epoch == cfg.training.num_epochs-1:
                    self.save_checkpoint()
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        
        # NEW: Final attention statistics summary
        if self.log_attention_stats:
            print("\n========= Attention Training Summary =========")
            if attention_stats['obs_lengths_used']:
                print(f"Observations lengths used during training:")
                print(f"  Average: {np.mean(attention_stats['obs_lengths_used']):.2f}")
                print(f"  Min: {np.min(attention_stats['obs_lengths_used'])}")
                print(f"  Max: {np.max(attention_stats['obs_lengths_used'])}")
                print(f"  Std: {np.std(attention_stats['obs_lengths_used']):.2f}")
            print(f"Final curriculum progress: {attention_stats['curriculum_progress']:.2%}")
            print("=============================================")
    
    def compute_loss(self, trajectory, noise, pred):
        pred_type = self.model.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def _print_dataset_diagnostics(self, cfg, dataset, train_dataloader, val_dataloaders):
        print()
        print("============= Attention Dataset Diagnostics =============")
        print(f"Number of datasets: {self.num_datasets}")
        print(f"Sample probabilities: {self.sample_probabilities}")
        print(f"[Training] Number of batches: {len(train_dataloader)}")
        
        # NEW: Print attention-specific info
        if hasattr(dataset, 'max_obs_steps') and hasattr(dataset, 'min_obs_steps'):
            print(f"Attention: max_obs_steps={dataset.max_obs_steps}, min_obs_steps={dataset.min_obs_steps}")
            print(f"Horizon: {dataset.horizon}")
            print(f"Variable obs curriculum: {getattr(dataset, 'attention_curriculum', 'N/A')}")
        
        for i in range(self.num_datasets):
            print(f"[Val {i}] Number of batches: {len(val_dataloaders[i])}")
        print()

        for i in range(self.num_datasets):
            val_dataset = dataset.get_validation_dataset(i)
            print(f"Dataset {i}: {dataset.zarr_paths[i]}")
            print("------------------------------------------------")
            print(f"Number of training demonstrations: {np.sum(dataset.train_masks[i])}")
            print(f"Number of validation demonstrations: {np.sum(dataset.val_masks[i])}")
            print(f"Number of training samples: {len(dataset.samplers[i])}")
            print(f"Number of validation samples: {len(val_dataset)}")
            print(f"Approx. number of training batches: {len(dataset.samplers[i]) // cfg.dataloader.batch_size}")
            print(f"Approx. number of validation batches: {len(val_dataset) // cfg.val_dataloader.batch_size}")
            print(f"Sample probability: {self.sample_probabilities[i]}")
            print()
        print("=========================================================")

    def _get_protected_paths(self, topk_manager_idx, topk_managers):
        """
        Returns the paths that should not be deleted by topk_manager
        """
        if len(topk_managers) == 1:
            return set()
        
        topk_manager = topk_managers[topk_manager_idx]

        protected_paths = set()
        for manager in topk_managers:
            protected_paths.update(manager.get_path_value_map().keys())
        
        # Remove the paths that can be deleted
        # If a ckpt is ONLY being tracked by topk_manager, it can be deleted
        for path in topk_manager.get_path_value_map().keys():
            protected = False
            for i, manager in enumerate(topk_managers):
                if i == topk_manager_idx:
                    continue
                if path in manager.get_path_value_map().keys():
                    protected = True
                    break
            if not protected:
                protected_paths.remove(path)
        
        return protected_paths
                

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionAttentionHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()