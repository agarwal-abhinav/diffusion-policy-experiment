import torch 

from torch.utils.data import DataLoader

import numpy as np 

import dill

import hydra

from diffusion_policy.common.pytorch_util import dict_apply

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

import copy

import math

# --- Helper functions ---

def gaussian_kernel(kernel_size=9, sigma=3, channels=3):
    """Create a Gaussian kernel for convolution."""
    # Create 1D Gaussian
    coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    # Create 2D Gaussian
    g2 = g[:, None] * g[None, :]
    kernel = g2.expand(channels, 1, kernel_size, kernel_size)
    return kernel

def low_pass_filter(x, kernel):
    """Apply low-pass (Gaussian blur) filter to input tensor x."""
    padding = kernel.shape[-1] // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])

def compute_cosine_similarity(z1, z2):
    """Compute average cosine similarity between two sets of latents."""
    sim = F.cosine_similarity(z1, z2, dim=1)
    mean = sim.mean().item()
    std = sim.std(unbiased=True).item()
    sem = std / (sim.numel() ** 0.5)
    return mean, std, sem

def center_gram(K):
    """Center a Gram matrix."""
    n = K.size(0)
    one_n = torch.ones((n, n), device=K.device) / n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

def linear_CKA(X, Y):
    """
    Compute linear CKA between two representations X, Y (shape [N, D]).
    """
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    K = X @ X.t()
    L = Y @ Y.t()
    Kc = center_gram(K)
    Lc = center_gram(L)
    return (Kc * Lc).sum() / torch.sqrt((Kc * Kc).sum() * (Lc * Lc).sum())

# CHECKPOINT_PATHS = ["data/outputs/context_length_exp_adam_data/1_obs/checkpoints/epoch=0095-val_loss=0.0366-val_ddim_mse=0.0006.ckpt", 
#                     "data/outputs/context_length_exp_adam_data/2_obs/checkpoints/epoch=0160-val_loss=0.0519-val_ddim_mse=0.0005.ckpt", 
#                     "data/outputs/context_length_exp_adam_data/5_obs/checkpoints/epoch=0135-val_loss=0.0207-val_ddim_mse=0.0002.ckpt",
#                      "data/outputs/context_length_exp_adam_data/10_obs_h_24/checkpoints/epoch=0085-val_loss=0.0186-val_ddim_mse=0.0003.ckpt",
#                     "data/outputs/context_length_exp_adam_data/12_obs/checkpoints/epoch=0155-val_loss=0.0244-val_ddim_mse=0.0002.ckpt", 
#                     "data/outputs/context_length_exp_adam_data/16_obs/checkpoints/epoch=0185-val_loss=0.0141-val_ddim_mse=0.0001.ckpt",]
# normalizer_paths = ["data/outputs/context_length_exp_adam_data/1_obs/normalizer.pt", 
#                     "data/outputs/context_length_exp_adam_data/2_obs/normalizer.pt", 
#                     "data/outputs/context_length_exp_adam_data/5_obs/normalizer.pt", 
#                     "data/outputs/context_length_exp_adam_data/10_obs_h_24/normalizer.pt", 
#                     "data/outputs/context_length_exp_adam_data/12_obs/normalizer.pt", 
#                     "data/outputs/context_length_exp_adam_data/16_obs/normalizer.pt"]

CHECKPOINT_PATHS = ["data/outputs/context_length_exp_adam_data_constant_model_size/1_obs/checkpoints/epoch=0175-val_loss=0.0887-val_ddim_mse=0.0016.ckpt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/2_obs/checkpoints/latest.ckpt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/5_obs/checkpoints/latest.ckpt",
                     "data/outputs/context_length_exp_adam_data_constant_model_size/10_obs/checkpoints/latest.ckpt",
                    "data/outputs/context_length_exp_adam_data_constant_model_size/12_obs/checkpoints/epoch=0140-val_loss=0.0235-val_ddim_mse=0.0001.ckpt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/16_obs/checkpoints/latest.ckpt",]
normalizer_paths = ["data/outputs/context_length_exp_adam_data_constant_model_size/1_obs/normalizer.pt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/2_obs/normalizer.pt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/5_obs/normalizer.pt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/10_obs/normalizer.pt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/12_obs/normalizer.pt", 
                    "data/outputs/context_length_exp_adam_data_constant_model_size/16_obs/normalizer.pt"]

def run(): 
    device = "cuda:0"

    kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3).to(device)

    low_freq_meta_cam_1 = []
    low_freq_meta_cam_2 = []
    high_freq_meta_cam_1 = []
    high_freq_meta_cam_2 = []

    low_freq_sem_meta_cam_1 = []
    low_freq_sem_meta_cam_2 = []
    high_freq_sem_meta_cam_1 = []
    high_freq_sem_meta_cam_2 = []

    for CHECKPOINT_PATH, normalizer_path in zip(CHECKPOINT_PATHS, normalizer_paths):
        print("start")
        payload = torch.load(open(CHECKPOINT_PATH, "rb"), pickle_module=dill)
        payload_cfg = payload["cfg"]

        cls = hydra.utils.get_class(payload_cfg._target_)
        workspace = cls(payload_cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        normalizer = torch.load(normalizer_path, weights_only=False)

        policy = workspace.model 
        policy.set_normalizer(normalizer)
        if payload_cfg.training.use_ema: 
            policy = workspace.ema_model 
            policy.set_normalizer(normalizer)
        policy.eval()
        policy.to(device)
        
        # payload_cfg.dataloader.batch_size = 200
        dataset = hydra.utils.instantiate(payload_cfg.task.dataset)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **payload_cfg.val_dataloader)
        print(payload_cfg.val_dataloader.batch_size)

        train_dataloader = DataLoader(dataset, **payload_cfg.dataloader)

        low_freq_cam_1_this = []
        low_freq_cam_2_this = []
        high_freq_cam_1_this = []
        high_freq_cam_2_this = []

        low_freq_sem_cam_1_this = []
        low_freq_sem_cam_2_this = []
        high_freq_sem_cam_1_this = []
        high_freq_sem_cam_2_this = []

        num_lists = policy.n_obs_steps 

        batch_num = 0
        for batch in val_dataloader: 
            B, _, C, H, W = batch["obs"]["overhead_camera"].shape
            N = policy.n_obs_steps

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            original_prop = batch["obs"]["agent_pos"][:, :policy.n_obs_steps, ...]
            original_cam_1 = batch["obs"]["overhead_camera"][:, :policy.n_obs_steps, ...].reshape(B*N, C, H, W)
            original_cam_2 = batch["obs"]["wrist_camera"][:, :policy.n_obs_steps, ...].reshape(B*N, C, H, W)

            kernel = kernel.to(device=device, dtype=original_cam_1.dtype)

            lp_cam_1 = low_pass_filter(original_cam_1, kernel)
            lp_cam_2 = low_pass_filter(original_cam_2, kernel)

            hp_cam_1 = original_cam_1 - lp_cam_1
            hp_cam_2 = original_cam_2 - lp_cam_2

            hp_cam_1_min = hp_cam_1.amin(dim=(2,3), keepdim=True)
            hp_cam_1_max = hp_cam_1.amax(dim=(2,3), keepdim=True)
            hp_cam_2_min = hp_cam_2.amin(dim=(2,3), keepdim=True)
            hp_cam_2_max = hp_cam_2.amax(dim=(2,3), keepdim=True)

            hp_cam_1 = (hp_cam_1 - hp_cam_1_min) / (hp_cam_1_max - hp_cam_1_min + 1e-8)
            hp_cam_2 = (hp_cam_2 - hp_cam_2_min) / (hp_cam_2_max - hp_cam_2_min + 1e-8)
            hp_cam_1 = hp_cam_1.clamp(0.0, 1.0)
            hp_cam_2 = hp_cam_2.clamp(0.0, 1.0)

            batch_hp = copy.deepcopy(batch)
            batch_lp = copy.deepcopy(batch)

            batch_hp["obs"]["overhead_camera"][:, :N, ...] = hp_cam_1.reshape(B, N, C, H, W)
            batch_hp["obs"]["wrist_camera"][:, :N, ...] = hp_cam_2.reshape(B, N, C, H, W)

            batch_lp["obs"]["overhead_camera"][:, :N, ...] = lp_cam_1.reshape(B, N, C, H, W)
            batch_lp["obs"]["wrist_camera"][:, :N, ...] = lp_cam_2.reshape(B, N, C, H, W)

            original_cam_1 = original_cam_1.reshape(B, N, C, H, W)
            original_cam_2 = original_cam_2.reshape(B, N, C, H, W)

            lp_cam_1 = lp_cam_1.reshape(B, N, C, H, W)
            lp_cam_2 = lp_cam_2.reshape(B, N, C, H, W)

            hp_cam_1 = hp_cam_1.reshape(B, N, C, H, W)
            hp_cam_2 = hp_cam_2.reshape(B, N, C, H, W)

            with torch.no_grad(): 
                reshaped, original = policy.get_encoder_output(batch)
                reshaped_lp, _ = policy.get_encoder_output(batch_lp)
                reshaped_hp, _ = policy.get_encoder_output(batch_hp)

                reshaped = reshaped.reshape(reshaped.shape[0], policy.n_obs_steps, 131)
                reshaped_lp = reshaped_lp.reshape(reshaped_lp.shape[0], policy.n_obs_steps, 131)
                reshaped_hp = reshaped_hp.reshape(reshaped_hp.shape[0], policy.n_obs_steps, 131)
                
            prop = reshaped[:, :, :3]
            z_cam_1 = reshaped[:, :, 3:67]
            z_cam_2 = reshaped[:, :, 67:131]

            prop_lp = reshaped_lp[:, :, :3]
            z_cam_1_lp = reshaped_lp[:, :, 3:67]
            z_cam_2_lp = reshaped_lp[:, :, 67:131]

            prop_hp = reshaped_hp[:, :, :3]
            z_cam_1_hp = reshaped_hp[:, :, 3:67]
            z_cam_2_hp = reshaped_hp[:, :, 67:131]
            
            # process things separately for each timesteps 

            for j in range(policy.n_obs_steps):
                original_cam_1_j = original_cam_1[:, j, ...]
                original_cam_2_j = original_cam_2[:, j, ...]

                lp_cam_1_j = lp_cam_1[:, j, ...]
                lp_cam_2_j = lp_cam_2[:, j, ...]

                hp_cam_1_j = hp_cam_1[:, j, ...]
                hp_cam_2_j = hp_cam_2[:, j, ...]

                z_orig_cam_1_j = z_cam_1[:, j, ...]
                z_orig_cam_2_j = z_cam_2[:, j, ...]

                z_lp_cam_1_j = z_cam_1_lp[:, j, ...]
                z_lp_cam_2_j = z_cam_2_lp[:, j, ...]

                z_hp_cam_1_j = z_cam_1_hp[:, j, ...]
                z_hp_cam_2_j = z_cam_2_hp[:, j, ...]

                cos_lp_cam_1, _, cos_lp_cam_1_sem = compute_cosine_similarity(z_orig_cam_1_j, z_lp_cam_1_j)
                cos_lp_cam_2, _, cos_lp_cam_2_sem = compute_cosine_similarity(z_orig_cam_2_j, z_lp_cam_2_j)

                cos_hp_cam_1, _, cos_hp_cam_1_sem = compute_cosine_similarity(z_orig_cam_1_j, z_hp_cam_1_j)
                cos_hp_cam_2, _, cos_hp_cam_2_sem = compute_cosine_similarity(z_orig_cam_2_j, z_hp_cam_2_j)

                if batch_num == 0: 
                    low_freq_cam_1_this.append(cos_lp_cam_1)
                    low_freq_cam_2_this.append(cos_lp_cam_2)
                    high_freq_cam_1_this.append(cos_hp_cam_1)
                    high_freq_cam_2_this.append(cos_hp_cam_2)

                    low_freq_sem_cam_1_this.append(cos_lp_cam_1_sem)
                    low_freq_sem_cam_2_this.append(cos_lp_cam_2_sem)
                    high_freq_sem_cam_1_this.append(cos_hp_cam_1_sem)
                    high_freq_sem_cam_2_this.append(cos_hp_cam_2_sem)
                else: 
                    temp_low_freq_cam_1 = low_freq_cam_1_this[j]
                    temp_low_freq_cam_2 = low_freq_cam_2_this[j]
                    temp_high_freq_cam_1 = high_freq_cam_1_this[j]
                    temp_high_freq_cam_2 = high_freq_cam_2_this[j]

                    temp_low_freq_sem_cam_1 = low_freq_sem_cam_1_this[j]
                    temp_low_freq_sem_cam_2 = low_freq_sem_cam_2_this[j]
                    temp_high_freq_sem_cam_1 = high_freq_sem_cam_1_this[j]
                    temp_high_freq_sem_cam_2 = high_freq_sem_cam_2_this[j]

                    low_freq_cam_1_this[j] = (temp_low_freq_cam_1 * batch_num * 64 + cos_lp_cam_1 * 64) / ((batch_num + 1) * 64)
                    low_freq_cam_2_this[j] = (temp_low_freq_cam_2 * batch_num * 64 + cos_lp_cam_2 * 64) / ((batch_num + 1) * 64)
                    high_freq_cam_1_this[j] = (temp_high_freq_cam_1 * batch_num * 64 + cos_hp_cam_1 * 64) / ((batch_num + 1) * 64)
                    high_freq_cam_2_this[j] = (temp_high_freq_cam_2 * batch_num * 64 + cos_hp_cam_2 * 64) / ((batch_num + 1) * 64)

                    low_freq_sem_cam_1_this[j] = math.sqrt((batch_num*64)/((batch_num+1)*64) * temp_low_freq_sem_cam_1**2 + \
                                                           (64)/((batch_num+1)*64) * cos_lp_cam_1_sem**2)
                    low_freq_sem_cam_2_this[j] = math.sqrt((batch_num*64)/((batch_num+1)*64) * temp_low_freq_sem_cam_2**2 + \
                                                           (64)/((batch_num+1)*64) * cos_lp_cam_2_sem**2)
                    high_freq_sem_cam_1_this[j] = math.sqrt((batch_num*64)/((batch_num+1)*64) * temp_high_freq_sem_cam_1**2 + \
                                                           (64)/((batch_num+1)*64) * cos_hp_cam_1_sem**2)
                    high_freq_sem_cam_2_this[j] = math.sqrt((batch_num*64)/((batch_num+1)*64) * temp_high_freq_sem_cam_2**2 + \
                                                           (64)/((batch_num+1)*64) * cos_hp_cam_2_sem**2)


                # breakpoint()
            batch_num += 1
            if batch_num == 20:
                break 

        low_freq_meta_cam_1.append(low_freq_cam_1_this)
        low_freq_meta_cam_2.append(low_freq_cam_2_this)
        high_freq_meta_cam_1.append(high_freq_cam_1_this)
        high_freq_meta_cam_2.append(high_freq_cam_2_this)
        low_freq_sem_meta_cam_1.append(low_freq_sem_cam_1_this)
        low_freq_sem_meta_cam_2.append(low_freq_sem_cam_2_this)
        high_freq_sem_meta_cam_1.append(high_freq_sem_cam_1_this)
        high_freq_sem_meta_cam_2.append(high_freq_sem_cam_2_this)

    # import pickle
    # with open("low_freq_meta_cam_1_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(low_freq_meta_cam_1, f)

    # with open("low_freq_meta_cam_2_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(low_freq_meta_cam_2, f)

    # with open("high_freq_meta_cam_1_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(high_freq_meta_cam_1, f)

    # with open("high_freq_meta_cam_2_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(high_freq_meta_cam_2, f)

    # with open("low_freq_sem_meta_cam_1_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(low_freq_sem_meta_cam_1, f)

    # with open("low_freq_sem_meta_cam_2_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(low_freq_sem_meta_cam_2, f)

    # with open("high_freq_sem_meta_cam_1_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(high_freq_sem_meta_cam_1, f)

    # with open("high_freq_sem_meta_cam_2_const_model_large_ho.pkl", "wb") as f:
    #     pickle.dump(high_freq_sem_meta_cam_2, f)

if __name__ == "__main__":
    run()

# # --- Main pipeline ---

# # Assume `images` is your tensor of shape [N, 3, 128, 128], values in [0, 255]
# # and `encoder` is your pretrained model mapping [B,3,128,128] -> [B, latent_dim]
# images = ...        # placeholder: load your images here
# encoder = ...       # placeholder: your encoder model
# mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)  # example
# std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# # Normalize to [0,1]
# imgs_norm = images.float() / 255.0

# # Create Gaussian kernel
# kernel = gaussian_kernel(kernel_size=9, sigma=3, channels=3).to(images.device)

# # Generate low-pass and high-pass versions
# imgs_lp = low_pass_filter(imgs_norm, kernel)
# imgs_hp = imgs_norm - imgs_lp

# # Apply the same normalizer
# imgs_orig_norm = (imgs_norm - mean) / std
# imgs_lp_norm   = (imgs_lp   - mean) / std
# imgs_hp_norm   = (imgs_hp   - mean) / std

# # Encode
# with torch.no_grad():
#     z_orig = encoder(imgs_orig_norm)
#     z_lp   = encoder(imgs_lp_norm)
#     z_hp   = encoder(imgs_hp_norm)

# # Compute cosine similarities
# cos_lp = compute_cosine_similarity(z_orig, z_lp)
# cos_hp = compute_cosine_similarity(z_orig, z_hp)
# print(f"Average cosine similarity: orig vs low-pass = {cos_lp:.4f}")
# print(f"Average cosine similarity: orig vs high-pass = {cos_hp:.4f}")

# # Compute CKA
# cka_lp = linear_CKA(z_orig, z_lp).item()
# cka_hp = linear_CKA(z_orig, z_hp).item()
# print(f"CKA: orig vs low-pass = {cka_lp:.4f}")
# print(f"CKA: orig vs high-pass = {cka_hp:.4f}")

# # PCA scatter
# all_z = torch.cat([z_orig, z_lp, z_hp], dim=0).cpu().numpy()
# labels = ['orig']*z_orig.shape[0] + ['low-pass']*z_lp.shape[0] + ['high-pass']*z_hp.shape[0]
# pca = PCA(n_components=2).fit_transform(all_z)
# plt.figure(figsize=(6,6))
# for label in ['orig', 'low-pass', 'high-pass']:
#     idx = [i for i, l in enumerate(labels) if l == label]
#     plt.scatter(pca[idx,0], pca[idx,1], label=label, alpha=0.5)
# plt.legend()
# plt.title('PCA of Latents')
# plt.xlabel('PC1'); plt.ylabel('PC2')
# plt.show()

# # Per-channel histogram for a chosen channel (e.g., channel 0)
# channel_idx = 0
# plt.figure()
# plt.hist(z_orig[:, channel_idx].cpu().numpy(), bins=30, alpha=0.5, label='orig')
# plt.hist(z_lp[:,   channel_idx].cpu().numpy(), bins=30, alpha=0.5, label='low-pass')
# plt.hist(z_hp[:,   channel_idx].cpu().numpy(), bins=30, alpha=0.5, label='high-pass')
# plt.legend()
# plt.title(f'Channel {channel_idx} Activation Distributions')
# plt.show()

# # (Optional) Frequency-response with sinusoidal gratings
# # Generate gratings at various frequencies and encode
# freqs = torch.linspace(0.01, 0.5, steps=10)
# gains = []
# for f in freqs:
#     # create grating pattern
#     x = torch.linspace(0, 1, steps=128)
#     X, Y = torch.meshgrid(x, x, indexing='ij')
#     grating = torch.sin(2 * np.pi * f * (X + Y))[None, None, ...].repeat(1,3,1,1)  # shape [1,3,128,128]
#     grating_norm = (grating - mean) / std
#     with torch.no_grad():
#         z_g = encoder(grating_norm.to(images.device))
#     gains.append(z_g.norm().item())

# plt.figure()
# plt.plot(freqs.numpy(), gains)
# plt.title('Encoder Gain vs Spatial Frequency')
# plt.xlabel('Cycles per pixel')
# plt.ylabel('Latent Norm')
# plt.show()
