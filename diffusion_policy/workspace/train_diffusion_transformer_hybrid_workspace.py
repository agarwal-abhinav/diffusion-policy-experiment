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
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
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

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

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

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

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
        train_sampling_batch = None

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
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
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

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

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
                            with tqdm.tqdm(val_dataloader, desc=f"Dataset {dataset_idx}Validation epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    if val_sampling_batches[dataset_idx] is None: 
                                        val_sampling_batches[dataset_idx] = batch
                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                                    
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                val_loss_per_dataset.append(val_loss)
                                # log epoch average validation loss
                                step_log[f'val_loss_{dataset_idx}'] = val_loss

                        overall_val_loss = 0
                        for i in range(self.num_datasets): 
                            overall_val_loss += self.sample_probabilities[i] * val_loss_per_dataset[i]
                        step_log['val_loss'] = overall_val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and cfg.training.log_val_mse:
                    with torch.no_grad():
                        val_ddpm_action_mses = []
                        val_ddim_action_mses = []
                        for dataset_idx in range(self.num_datasets): 
                            val_sampling_batch = val_sampling_batches[dataset_idx]
                            val_batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            val_obs_dict = {key: val_batch[key] for key in val_batch.keys() if key != 'action'}
                            val_gt_action = val_sampling_batch['action']

                            if cfg.training.eval_mse_DDPM: 
                                result = policy.predict_action(val_obs_dict, use_DDIM=False)
                                pred_action = result['action_pred']
                                mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                                step_log[f'val_ddpm_mse_{dataset_idx}'] = mse.item()
                                val_ddpm_action_mses.append(mse.item())
                            
                            if cfg.training.eval_mse_DDIM: 
                                result = policy.predict_action(val_obs_dict, use_DDIM=True)
                                pred_action = result['action_pred']
                                mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
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

                        # # sample trajectory from training set, and evaluate difference
                        # batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        # obs_dict = batch['obs']
                        # gt_action = batch['action']
                        
                        # result = policy.predict_action(obs_dict)
                        # pred_action = result['action_pred']
                        # mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # step_log['train_action_mse_error'] = mse.item()
                        # del batch
                        # del obs_dict
                        # del gt_action
                        # del result
                        # del pred_action
                        # del mse
                
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
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
