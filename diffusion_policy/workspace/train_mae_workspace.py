"""
Workspace for MAE (Masked Autoencoder) pre-training of ViT encoders.

Trains a ViT encoder via masked image reconstruction on robot observation
images from zarr replay buffers. After training, exports encoder weights
as a .pth file loadable by the downstream policy via vit_pretrained.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import copy
import random
import pathlib
import hydra
import torch
import numpy as np
import wandb
import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.dataset.mae_image_dataset import MAEImageDataset

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainMAEWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Create MAE model
        self.model = hydra.utils.instantiate(cfg.mae_model)
        device = torch.device(cfg.training.device)
        self.model = self.model.to(device)

        n_params = sum(p.numel() for p in self.model.parameters())
        n_encoder = sum(p.numel() for p in self.model.encoder.parameters())
        n_decoder = n_params - n_encoder
        print(f"MAE params: encoder={n_encoder:,}, decoder={n_decoder:,}, total={n_params:,}")

        # EMA
        self.ema_model = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Optimizer
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Mixed precision
        self.use_amp = getattr(cfg.training, 'use_amp', False)
        self.scaler = GradScaler("cuda") if self.use_amp else None
        print(f"Mixed precision: {'enabled' if self.use_amp else 'disabled'}")

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Checkpoint manager
        ckpt_save_dir = os.path.join(self.output_dir, 'checkpoints')
        topk_manager = TopKCheckpointManager(
            save_dir=ckpt_save_dir,
            **cfg.checkpoint.topk
        )

        # Resume
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                self.epoch += 1

        # Dataset
        dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
        train_dataset, val_dataset = MAEImageDataset.create_train_val(
            **dataset_cfg)
        train_dataloader = DataLoader(train_dataset, **cfg.dataloader)
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=len(train_dataloader) * cfg.training.num_epochs,
            last_epoch=self.global_step - 1,
        )

        # EMA
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # wandb
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        # Device
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Debug mode
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.val_every = 1
            cfg.training.checkpoint_every = 1

        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        resume_epoch = self.epoch
        log_reconstructions_every = getattr(
            cfg.training, 'log_reconstructions_every', 20)

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                epoch = local_epoch_idx + resume_epoch
                self.epoch = epoch

                # --- Training ---
                self.model.train()
                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Train epoch {epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, imgs in enumerate(tepoch):
                        imgs = imgs.to(device, non_blocking=True)

                        if self.use_amp:
                            with autocast("cuda", dtype=torch.float16):
                                loss, pred, mask = self.model(imgs)
                        else:
                            loss, pred, mask = self.model(imgs)

                        # Backward
                        self.optimizer.zero_grad()
                        if self.use_amp:
                            self.scaler.scale(loss).backward()
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            loss.backward()
                            self.optimizer.step()

                        lr_scheduler.step()

                        # EMA update
                        if ema is not None:
                            ema.step(self.model)

                        train_losses.append(loss.item())
                        self.global_step += 1
                        tepoch.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

                        max_steps = getattr(cfg.training, 'max_train_steps', None)
                        if max_steps is not None and batch_idx >= max_steps:
                            break

                avg_train_loss = np.mean(train_losses)

                # --- Validation ---
                val_loss = None
                if (epoch % cfg.training.val_every) == 0:
                    eval_model = self.ema_model if ema is not None else self.model
                    eval_model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for batch_idx, imgs in enumerate(val_dataloader):
                            imgs = imgs.to(device, non_blocking=True)
                            loss, pred, mask = eval_model(imgs)
                            val_losses.append(loss.item())
                            max_val = getattr(cfg.training, 'max_val_steps', None)
                            if max_val is not None and batch_idx >= max_val:
                                break
                    val_loss = np.mean(val_losses)

                # --- Logging ---
                step_log = {
                    'train_loss': avg_train_loss,
                    'lr': lr_scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'global_step': self.global_step,
                }
                if val_loss is not None:
                    step_log['val_loss'] = val_loss

                # Log reconstruction images
                if epoch % log_reconstructions_every == 0:
                    self._log_reconstructions(eval_model if ema else self.model,
                                              val_dataloader, device, step_log)

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)

                # --- Checkpointing ---
                if (epoch % cfg.training.checkpoint_every) == 0 and val_loss is not None:
                    # TopK checkpoint
                    metric_dict = {
                        'val_loss': val_loss,
                        'epoch': epoch,
                        'global_step': self.global_step,
                    }
                    ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if ckpt_path is not None:
                        self.save_checkpoint(path=ckpt_path)

                    # Always save latest
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()

                print(f"Epoch {epoch}: train_loss={avg_train_loss:.6f}"
                      + (f", val_loss={val_loss:.6f}" if val_loss is not None else ""))

        # Export encoder weights at the end
        export_path = os.path.join(self.output_dir, 'mae_encoder.pth')
        self.export_encoder_weights(export_path)

        # Also export from EMA model if available
        if ema is not None:
            ema_export_path = os.path.join(self.output_dir, 'mae_encoder_ema.pth')
            self.export_encoder_weights(ema_export_path, use_ema=True)

    def export_encoder_weights(self, path, use_ema=False):
        """Export ViT encoder state_dict to a .pth file."""
        model = self.ema_model if use_ema else self.model
        # Handle DataParallel wrapper
        if hasattr(model, 'module'):
            model = model.module
        state_dict = model.get_encoder_state_dict()
        torch.save(state_dict, path)
        print(f"Exported encoder weights ({len(state_dict)} keys) to {path}")

    @torch.no_grad()
    def _log_reconstructions(self, model, val_dataloader, device, step_log,
                             n_images=4):
        """Log original / masked / reconstructed images to wandb."""
        model.eval()
        imgs = next(iter(val_dataloader))[:n_images].to(device)
        loss, pred, mask = model(imgs)

        # Denormalize for visualization (undo ImageNet norm approximately)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape(1, 3, 1, 1)
        imgs_vis = (imgs * std + mean).clamp(0, 1)

        # Reconstruct full image from pred
        if hasattr(model, 'module'):
            m = model.module
        else:
            m = model
        recon = m.unpatchify(pred)
        # Pred was computed on normalized targets if norm_pix_loss;
        # for visualization, just denormalize the original patches
        # and replace masked patches with prediction (approximate)
        target_patches = m.patchify(imgs)
        # Blend: use original for visible, pred for masked
        mask_img = mask.unsqueeze(-1).expand_as(target_patches)
        blended = target_patches * (1 - mask_img) + pred * mask_img
        blended_img = m.unpatchify(blended)
        blended_vis = (blended_img * std + mean).clamp(0, 1)

        # Create masked visualization
        mask_expanded = mask.unsqueeze(-1).expand_as(target_patches)
        masked_patches = target_patches * (1 - mask_expanded)
        masked_img = m.unpatchify(masked_patches)
        masked_vis = (masked_img * std + mean).clamp(0, 1)

        # Log to wandb
        import torchvision
        grid_orig = torchvision.utils.make_grid(imgs_vis, nrow=n_images)
        grid_masked = torchvision.utils.make_grid(masked_vis, nrow=n_images)
        grid_recon = torchvision.utils.make_grid(blended_vis, nrow=n_images)

        step_log['reconstructions/original'] = wandb.Image(
            grid_orig.permute(1, 2, 0).cpu().numpy())
        step_log['reconstructions/masked'] = wandb.Image(
            grid_masked.permute(1, 2, 0).cpu().numpy())
        step_log['reconstructions/reconstructed'] = wandb.Image(
            grid_recon.permute(1, 2, 0).cpu().numpy())
