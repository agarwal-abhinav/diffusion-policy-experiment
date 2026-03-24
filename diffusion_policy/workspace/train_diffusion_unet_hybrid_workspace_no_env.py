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
from torch.amp import GradScaler, autocast
from einops import rearrange, reduce
import torch.nn.functional as F
from diffusion_policy.workspace.base_workspace import BaseWorkspace
# from diffusion_policy.policy.diffusion_unet_hybrid_image_targeted_policy import DiffusionUnetHybridImageTargetedPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
# from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager, IntervalCheckpointManager, CheckpointManagers
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
        
class TrainDiffusionUnetHybridWorkspaceNoEnv(BaseWorkspace):
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

        print(f"Running on {num_GPU} GPU(s).")
        self.model = DataParallelWrapper(self.model, device_ids=range(num_GPU))

        # load pretrained model if finetuning
        if 'pretrained_checkpoint' in cfg and cfg.pretrained_checkpoint is not None:
            print(f"Loading pretrained model from {cfg.pretrained_checkpoint}.")
            path = pathlib.Path(cfg.pretrained_checkpoint)
            payload = torch.load(path.open('rb'), pickle_module=dill)
            self.model.load_state_dict(payload['state_dicts']['model'])
        else:
            print("Initializing model using default parameters.")

        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state with variable learning rates
        self.encoder_lr_scale = getattr(cfg.training, 'encoder_lr_scale', 1.0)
        self.use_variable_lr = getattr(cfg.training, 'use_variable_lr', False)
        
        if self.use_variable_lr and hasattr(self.model, 'obs_encoder'):
            obs_enc = self.model.obs_encoder
            base_lr = cfg.optimizer.lr
            enc_lr = base_lr * self.encoder_lr_scale

            # Collect ViT backbone params (low LR) vs temporal/other params (full LR)
            vit_params = []
            if hasattr(obs_enc, 'vit_encoders'):
                seen_ids = set()
                for key in obs_enc.rgb_keys:
                    vit = obs_enc.vit_encoders[key]
                    if hasattr(vit, 'temporal_layers'):
                        # VideoViTEncoder (D/E): only spatial parts get low LR
                        for p in vit.patch_embed.parameters():
                            if id(p) not in seen_ids:
                                vit_params.append(p); seen_ids.add(id(p))
                        if id(vit.cls_token) not in seen_ids:
                            vit_params.append(vit.cls_token); seen_ids.add(id(vit.cls_token))
                        if id(vit.pos_embed) not in seen_ids:
                            vit_params.append(vit.pos_embed); seen_ids.add(id(vit.pos_embed))
                        for p in vit.blocks.parameters():
                            if id(p) not in seen_ids:
                                vit_params.append(p); seen_ids.add(id(p))
                        for p in vit.norm.parameters():
                            if id(p) not in seen_ids:
                                vit_params.append(p); seen_ids.add(id(p))
                    else:
                        # Standard timm ViT (A/B/C): all params get low LR
                        for p in vit.parameters():
                            if id(p) not in seen_ids:
                                vit_params.append(p); seen_ids.add(id(p))

            vit_param_ids = {id(p) for p in vit_params}
            other_params = [p for p in self.model.parameters()
                           if id(p) not in vit_param_ids]

            n_vit = sum(p.numel() for p in vit_params if p.requires_grad)
            n_other = sum(p.numel() for p in other_params if p.requires_grad)
            print(f"Variable LR: ViT backbone ({n_vit:,} params) LR={enc_lr:.6f}, "
                  f"rest ({n_other:,} params) LR={base_lr:.6f}")

            param_groups = [
                {'params': vit_params, 'lr': enc_lr},
                {'params': other_params, 'lr': base_lr}
            ]

            import torch.optim as optim
            optimizer_class = getattr(optim, cfg.optimizer._target_.split('.')[-1])
            optimizer_kwargs = dict(cfg.optimizer)
            optimizer_kwargs.pop('_target_')
            optimizer_kwargs.pop('lr')
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
        self.scaler = GradScaler("cuda") if self.use_amp else None
        print(f"Mixed precision training: {'enabled' if self.use_amp else 'disabled'}")

    def _should_precompute_vit(self, cfg):
        """Check if we should pre-compute frozen ViT embeddings."""
        vit_freeze = getattr(cfg.policy, 'vit_freeze', False)
        variant = getattr(cfg.policy, 'vit_variant', None)
        if not vit_freeze or variant is None:
            return False
        if variant in ('D', 'E'):
            return False
        return True

    def _precompute_or_load_vit_cache(self, cfg, dataset):
        """
        Pre-compute ViT embeddings or load from disk cache (center crop).
        Returns (cache_dict, embed_dim).
        """
        import hashlib

        # Build cache key from model config + data paths
        use_spatial = getattr(cfg.policy, 'use_spatial_softmax', False)
        cache_key = (
            f"{cfg.policy.vit_model_name}_"
            f"{cfg.policy.vit_pretrained}_"
            f"{tuple(cfg.policy.crop_shape)}_"
            f"spatial={use_spatial}_"
            f"{sorted(dataset.zarr_paths)}"
        )
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_dir = os.path.join(self.output_dir, 'vit_cache')
        cache_path = os.path.join(cache_dir, f'vit_embeddings_{cache_hash}.pt')

        # Try loading from disk
        if os.path.exists(cache_path):
            print(f"Loading ViT embedding cache from {cache_path}")
            cached = torch.load(cache_path, map_location='cpu')
            print(f"  Loaded cache with embed_dim={cached['embed_dim']}")
            return cached['cache'], cached['embed_dim']

        # Pre-compute (center crop — eval mode)
        print("Pre-computing frozen ViT embeddings for all frames...")
        obs_encoder = self.model.obs_encoder
        cache, embed_dim = obs_encoder.precompute_all_embeddings(
            dataset.replay_buffers, batch_size=128, train_mode=False)

        # Save to disk
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({'cache': cache, 'embed_dim': embed_dim}, cache_path)
        print(f"  Saved ViT embedding cache to {cache_path}")

        return cache, embed_dim

    def _recompute_vit_cache(self, dataset):
        """
        Recompute ViT embeddings with random crops (train mode).
        Called at the start of each epoch when random_crop_cache is enabled.
        Returns (cache_dict, embed_dim).
        """
        obs_encoder = self.model.obs_encoder
        cache, embed_dim = obs_encoder.precompute_all_embeddings(
            dataset.replay_buffers, batch_size=128, train_mode=True)
        return cache, embed_dim

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # configure checkpoint managers BEFORE resume so their state
        # gets restored from the checkpoint (backward compatible —
        # old checkpoints without manager state are handled gracefully)
        ckpt_save_dir = os.path.join(self.output_dir, 'checkpoints')
        if not isinstance(cfg.checkpoint, ListConfig):
            topk_managers = [TopKCheckpointManager(
                save_dir=ckpt_save_dir,
                **cfg.checkpoint.topk
            )]
            save_last_ckpt = cfg.checkpoint.save_last_ckpt
            save_last_snapshot = cfg.checkpoint.save_last_snapshot
        else:
            topk_managers = []
            save_last_ckpt = False
            save_last_snapshot = False
            for ckpt_cfg in cfg.checkpoint:
                topk_managers.append(TopKCheckpointManager(
                    save_dir=ckpt_save_dir,
                    **ckpt_cfg.topk
                ))
                save_last_ckpt = save_last_ckpt or ckpt_cfg.save_last_ckpt
                save_last_snapshot = save_last_snapshot or ckpt_cfg.save_last_snapshot

        # interval checkpoint manager (optional, backward compatible)
        # Configured via top-level 'checkpoint_interval' key in config
        interval_cfg = getattr(cfg, 'checkpoint_interval', None)

        interval_manager = None
        if interval_cfg is not None:
            total_steps = getattr(cfg.training, 'total_train_steps', None)
            if total_steps is None:
                print("Warning: interval checkpoints require total_train_steps in config. Skipping.")
            else:
                interval_manager = IntervalCheckpointManager(
                    save_dir=ckpt_save_dir,
                    total_training_steps=total_steps,
                    **interval_cfg,
                )
                print(f"Interval checkpoints: {interval_manager.num_checkpoints} "
                      f"checkpoints over last {interval_manager.last_n_steps} steps "
                      f"(at steps {sorted(interval_manager.save_steps)})")

        # Store on self so BaseWorkspace.save_checkpoint auto-persists state
        self.checkpoint_managers = CheckpointManagers(
            topk_managers=topk_managers,
            interval_manager=interval_manager,
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # self.epoch is loaded with the last completed epoch
                # the current epoch is the next epoch (hence += 1)
                self.epoch += 1

        # configure dataset and save normalizer
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)

        # Pre-compute frozen ViT embeddings (skips ViT during training)
        self._use_random_crop_cache = False
        if self._should_precompute_vit(cfg):
            self._use_random_crop_cache = getattr(
                cfg.policy, 'random_crop_cache', False)
            # Always compute center-crop cache first (used by val datasets,
            # and as initial training cache if random_crop_cache is off)
            cache, embed_dim = self._precompute_or_load_vit_cache(cfg, dataset)
            dataset.set_embedding_cache(cache, embed_dim)
            self._vit_embed_dim_for_cache = embed_dim

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, os.path.join(self.output_dir, 'normalizer.pt'))

        # configure validation datasets
        # Val datasets get center-crop cache (set above via get_validation_dataset
        # propagation). They are never refreshed with random crops.
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

        from collections import defaultdict

        if getattr(cfg.policy, "rescale_encoder_gradients", False) == True:
            hooks_by_param = defaultdict(list)
            for name, p in self.model.obs_encoder.named_parameters():
                # getattr will return {} if no hooks were registered
                hooks = getattr(p, '_backward_hooks', {})
                if hooks:
                    hooks_by_param[name] = list(hooks.items())

            # Print a report
            for name, hook_list in hooks_by_param.items():
                print(f"Param {name} has {len(hook_list)} hook(s):")
                for hook_id, fn in hook_list:
                    print(f"  id={hook_id}, fn={fn}")

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

        # checkpoint config validation
        assert cfg.training.checkpoint_every % cfg.training.val_every == 0
        # Retrieve managers from self (created before resume, state may be restored)
        topk_managers = self.checkpoint_managers.topk_managers
        interval_manager = self.checkpoint_managers.interval_manager


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

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        resume_epoch = self.epoch
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(resume_epoch, cfg.training.num_epochs):
                step_log = dict()

                # Recompute ViT cache with fresh random crops each epoch
                if self._use_random_crop_cache:
                    cache, _ = self._recompute_vit_cache(dataset)
                    dataset.set_embedding_cache(cache, self._vit_embed_dim_for_cache)

                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if self.new_type_dataloader:
                            # this is done to maintain validity for older code
                            for key in dataset.rgb_keys:
                                if batch['obs'][key].ndim == 5:  # images (B,T,H,W,C), not embeddings
                                    batch['obs'][key] = torch.moveaxis(batch['obs'][key], -1, 2) / 255.0
                        batch_size = batch['action'].shape[0]

                        # Compute loss via policy (handles dynamic prediction horizon)
                        if self.use_amp:
                            with autocast("cuda", dtype=torch.float16):
                                raw_loss = self.model.compute_loss(batch)
                                loss = raw_loss / cfg.training.gradient_accumulate_every
                        else:
                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                        
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
                        if hasattr(dataset, 'set_training_step'):
                            dataset.set_training_step(self.global_step)
                            if hasattr(dataset, 'current_max'):
                                step_log['current_level'] = dataset.current_max

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
                        

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

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
                                        # this is done to maintain validity for older code
                                        for key in dataset.rgb_keys:
                                            if batch['obs'][key].ndim == 5:  # images (B,T,H,W,C), not embeddings
                                                batch['obs'][key] = torch.moveaxis(batch['obs'][key], -1, 2) / 255.0
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
                            val_gt_action = val_batch['action']

                            # Evaluate MSE when diffusing with DDPM
                            if cfg.training.eval_mse_DDPM:
                                if 'sample_metadata' in val_batch.keys():
                                    result = policy.predict_action(val_obs_dict, use_DDIM=False, num_obs_tokens=val_batch['sample_metadata']['num_obs_steps'])
                                else:
                                    result = policy.predict_action(val_obs_dict, use_DDIM=False)
                                # Compare full predicted trajectory against GT
                                pred_action = result['action_pred']
                                offset = result['action_pred_offset']
                                val_action = val_gt_action[:, offset:offset + pred_action.shape[1], :]
                                mse = torch.nn.functional.mse_loss(pred_action, val_action)
                                step_log[f'val_ddpm_mse_{dataset_idx}'] = mse.item()
                                val_ddpm_action_mses.append(mse.item())

                            # Evaluate MSE when diffusing with DDIM
                            if cfg.training.eval_mse_DDIM:
                                if 'sample_metadata' in val_batch.keys():
                                    result = policy.predict_action(val_obs_dict, use_DDIM=True, num_obs_tokens=val_batch['sample_metadata']['num_obs_steps'])
                                else:
                                    result = policy.predict_action(val_obs_dict, use_DDIM=True)
                                pred_action = result['action_pred']
                                offset = result['action_pred_offset']
                                val_action = val_gt_action[:, offset:offset + pred_action.shape[1], :]
                                mse = torch.nn.functional.mse_loss(pred_action, val_action)
                                step_log[f'val_ddim_mse_{dataset_idx}'] = mse.item()
                                val_ddim_action_mses.append(mse.item())
                        
                        # Compute weighted val action MSEs
                        if cfg.training.eval_mse_DDPM:
                            val_ = 0
                            for i in range(self.num_datasets):
                                val_ += self.sample_probabilities[i] * val_ddpm_action_mses[i]
                            step_log['val_ddpm_mse'] = val_
                        if cfg.training.eval_mse_DDIM:
                            val_ = 0
                            for i in range(self.num_datasets):
                                val_ += self.sample_probabilities[i] * val_ddim_action_mses[i]
                            step_log['val_ddim_mse'] = val_


                        del val_batch
                        del val_obs_dict
                        del val_gt_action
                        del result
                        del pred_action
                        del mse
                
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

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_paths = []
                    for i, topk_manager in enumerate(topk_managers):
                        protected_ckpts = self._get_protected_paths(i, topk_managers)
                        ckpt_path = topk_manager.get_ckpt_path(metric_dict, protected_ckpts)
                        topk_ckpt_paths.append(ckpt_path)

                    for i, topk_ckpt_path in enumerate(topk_ckpt_paths):
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                            break

                    # interval checkpoints (evenly spaced over last N steps)
                    if interval_manager is not None:
                        interval_path = interval_manager.get_ckpt_path(metric_dict)
                        if interval_path is not None:
                            self.save_checkpoint(path=interval_path)
                            print(f"Saved interval checkpoint at step {self.global_step}")
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
    
    def _extract_gt_action_window(self, val_gt_action, val_batch, policy):
        """Extract the n_action_steps ground truth window matching predict_action output."""
        n_past = getattr(policy, 'n_past_action_steps', None)
        n_action_steps = policy.n_action_steps
        if 'sample_metadata' in val_batch and n_past is not None:
            # Per-sample: To varies
            num_obs = val_batch['sample_metadata']['num_obs_steps']
            To = num_obs[0].item()  # assume uniform in val batch
            past_in_pred = min(n_past, To)
        else:
            To = getattr(policy, 'max_obs_steps', policy.n_obs_steps)
            past_in_pred = To
            if n_past is not None:
                past_in_pred = min(n_past, To)
        # In the full action tensor, current action is at To-1
        start = To - 1
        end = start + n_action_steps
        return val_gt_action[:, start:end]

    def _print_dataset_diagnostics(self, cfg, dataset, train_dataloader, val_dataloaders):
        print()
        print("============= Dataset Diagnostics =============")
        print(f"Number of datasets: {self.num_datasets}")
        print(f"Sample probabilities: {self.sample_probabilities}")
        print(f"[Training] Number of batches: {len(train_dataloader)}")
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
        print("================================================")

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
    workspace = TrainDiffusionUnetHybridWorkspaceNoEnv(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
