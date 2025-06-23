import torch 
from torch.utils.data import DataLoader
import numpy as np 
import dill
import hydra
from diffusion_policy.common.pytorch_util import dict_apply
import torch.nn.functional as F
import copy
import math
import pickle

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

# Configuration lists
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

    # Meta lists for cosine and CKA
    low_freq_meta_cam_1 = []
    low_freq_meta_cam_2 = []
    high_freq_meta_cam_1 = []
    high_freq_meta_cam_2 = []
    low_freq_cka_meta_cam_1 = []
    low_freq_cka_meta_cam_2 = []
    high_freq_cka_meta_cam_1 = []
    high_freq_cka_meta_cam_2 = []

    low_freq_sem_meta_cam_1 = []
    low_freq_sem_meta_cam_2 = []
    high_freq_sem_meta_cam_1 = []
    high_freq_sem_meta_cam_2 = []

    for ckpt_path, norm_path in zip(CHECKPOINT_PATHS, normalizer_paths):
        # Load model and data...
        print("start")
        payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)
        normalizer = torch.load(norm_path, weights_only=False)

        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.set_normalizer(normalizer)
        policy.eval().to(device)

        val_dataset = hydra.utils.instantiate(cfg.task.dataset).get_validation_dataset()
        val_loader = DataLoader(val_dataset, **cfg.val_dataloader)

        # Per-checkpoint storage
        lf_c1 = []
        lf_c2 = []
        hf_c1 = []
        hf_c2 = []
        lf_cka_c1 = []
        lf_cka_c2 = []
        hf_cka_c1 = []
        hf_cka_c2 = []
        lf_sem_c1 = []
        lf_sem_c2 = []
        hf_sem_c1 = []
        hf_sem_c2 = []

        batch_num = 0
        for batch in val_loader:
            print(str(batch_num))
            B, _, C, H, W = batch["obs"]["overhead_camera"].shape
            N = policy.n_obs_steps
            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

            # Prepare original, low-pass, high-pass for both cameras...
            oc1 = batch["obs"]["overhead_camera"][:, :N].reshape(B*N, C, H, W)
            oc2 = batch["obs"]["wrist_camera"][:, :N].reshape(B*N, C, H, W)

            kernel = kernel.to(device=device, dtype=oc1.dtype)

            lp1 = low_pass_filter(oc1, kernel)
            lp2 = low_pass_filter(oc2, kernel)
            hp1 = oc1 - lp1
            hp2 = oc2 - lp2
            # map hp to [0,1]
            for hp in [hp1, hp2]:
                mn = hp.amin((2,3), keepdim=True)
                mx = hp.amax((2,3), keepdim=True)
                hp.sub_(mn).div_(mx - mn + 1e-8).clamp_(0,1)

            # Build batches
            def make_batch(cam_variant, orig_batch):
                b = copy.deepcopy(batch)
                b["obs"]["overhead_camera"][:, :N] = cam_variant[0].reshape(B, N, C, H, W)
                b["obs"]["wrist_camera"][:, :N]    = cam_variant[1].reshape(B, N, C, H, W)
                return b

            # batch_lp = make_batch((lp1, lp2), batch)
            # batch_hp = make_batch((hp1, hp2), batch)

            batch_hp = copy.deepcopy(batch)
            batch_lp = copy.deepcopy(batch)

            batch_hp["obs"]["overhead_camera"][:, :N, ...] = hp1.reshape(B, N, C, H, W)
            batch_hp["obs"]["wrist_camera"][:, :N, ...] = hp2.reshape(B, N, C, H, W)

            batch_lp["obs"]["overhead_camera"][:, :N, ...] = lp1.reshape(B, N, C, H, W)
            batch_lp["obs"]["wrist_camera"][:, :N, ...] = lp2.reshape(B, N, C, H, W)

            oc1 = oc1.reshape(B, N, C, H, W)
            oc2 = oc2.reshape(B, N, C, H, W)

            lp1 = lp1.reshape(B, N, C, H, W)
            lp2 = lp2.reshape(B, N, C, H, W)

            hp1 = hp1.reshape(B, N, C, H, W)
            hp2 = hp2.reshape(B, N, C, H, W)

            with torch.no_grad(): 
                reshaped, original = policy.get_encoder_output(batch)
                reshaped_lp, _ = policy.get_encoder_output(batch_lp)
                reshaped_hp, _ = policy.get_encoder_output(batch_hp)

                reshaped = reshaped.reshape(reshaped.shape[0], policy.n_obs_steps, 131)
                reshaped_lp = reshaped_lp.reshape(reshaped_lp.shape[0], policy.n_obs_steps, 131)
                reshaped_hp = reshaped_hp.reshape(reshaped_hp.shape[0], policy.n_obs_steps, 131)


            # extract camera features
            prop = reshaped[:, :, :3]
            z_cam_1 = reshaped[:, :, 3:67]
            z_cam_2 = reshaped[:, :, 67:131]

            prop_lp = reshaped_lp[:, :, :3]
            z_cam_1_lp = reshaped_lp[:, :, 3:67]
            z_cam_2_lp = reshaped_lp[:, :, 67:131]

            prop_hp = reshaped_hp[:, :, :3]
            z_cam_1_hp = reshaped_hp[:, :, 3:67]
            z_cam_2_hp = reshaped_hp[:, :, 67:131]

            for j in range(N):
                z_orig_cam_1_j = z_cam_1[:, j, ...]
                z_orig_cam_2_j = z_cam_2[:, j, ...]

                z_lp_cam_1_j = z_cam_1_lp[:, j, ...]
                z_lp_cam_2_j = z_cam_2_lp[:, j, ...]

                z_hp_cam_1_j = z_cam_1_hp[:, j, ...]
                z_hp_cam_2_j = z_cam_2_hp[:, j, ...]

                # Cosine
                cl1, _, sem_l1 = compute_cosine_similarity(z_orig_cam_1_j, z_lp_cam_1_j)
                ch1, _, sem_h1 = compute_cosine_similarity(z_orig_cam_1_j, z_hp_cam_1_j)
                cl2, _, sem_l2 = compute_cosine_similarity(z_orig_cam_2_j, z_lp_cam_2_j)
                ch2, _, sem_h2 = compute_cosine_similarity(z_orig_cam_2_j, z_hp_cam_2_j)

                # CKA
                cka_l1 = linear_CKA(z_orig_cam_1_j, z_lp_cam_1_j).item()
                cka_h1 = linear_CKA(z_orig_cam_1_j, z_hp_cam_1_j).item()
                cka_l2 = linear_CKA(z_orig_cam_2_j, z_lp_cam_2_j).item()
                cka_h2 = linear_CKA(z_orig_cam_2_j, z_hp_cam_2_j).item()

                if batch_num == 0:
                    lf_c1.append(cl1); hf_c1.append(ch1)
                    lf_c2.append(cl2); hf_c2.append(ch2)
                    lf_cka_c1.append(cka_l1); hf_cka_c1.append(cka_h1)
                    lf_cka_c2.append(cka_l2); hf_cka_c2.append(cka_h2)
                    lf_sem_c1.append(sem_l1); hf_sem_c1.append(sem_h1)
                    lf_sem_c2.append(sem_l2); hf_sem_c2.append(sem_h2)
                else:
                    # running average for cosines and CKAs
                    def update(lst, new):
                        prev = lst[j]
                        total = batch_num * B
                        lst[j] = (prev * total + new * B) / ((batch_num+1) * B)

                    update(lf_c1, cl1); update(hf_c1, ch1)
                    update(lf_c2, cl2); update(hf_c2, ch2)
                    update(lf_cka_c1, cka_l1); update(hf_cka_c1, cka_h1)
                    update(lf_cka_c2, cka_l2); update(hf_cka_c2, cka_h2)

                    # SEM combination: similar to previously used formula
                    def update_sem(lst, sem_sq):
                        prev_sem = lst[j]
                        combined = math.sqrt((batch_num * B)/((batch_num+1)*B)*prev_sem**2 + (B)/((batch_num+1)*B)*sem_sq**2)
                        lst[j] = combined

                    update_sem(lf_sem_c1, sem_l1); update_sem(hf_sem_c1, sem_h1)
                    update_sem(lf_sem_c2, sem_l2); update_sem(hf_sem_c2, sem_h2)

            batch_num += 1
            if batch_num == 20:
                break

        # append per-checkpoint
        low_freq_meta_cam_1.append(lf_c1)
        low_freq_meta_cam_2.append(lf_c2)
        high_freq_meta_cam_1.append(hf_c1)
        high_freq_meta_cam_2.append(hf_c2)
        low_freq_cka_meta_cam_1.append(lf_cka_c1)
        low_freq_cka_meta_cam_2.append(lf_cka_c2)
        high_freq_cka_meta_cam_1.append(hf_cka_c1)
        high_freq_cka_meta_cam_2.append(hf_cka_c2)
        low_freq_sem_meta_cam_1.append(lf_sem_c1)
        low_freq_sem_meta_cam_2.append(lf_sem_c2)
        high_freq_sem_meta_cam_1.append(hf_sem_c1)
        high_freq_sem_meta_cam_2.append(hf_sem_c2)

    # Save all results
    # with open("low_freq_meta_cam_1.pkl", "wb") as f:
    #     pickle.dump(low_freq_meta_cam_1, f)
    # with open("low_freq_meta_cam_2.pkl", "wb") as f:
    #     pickle.dump(low_freq_meta_cam_2, f)
    # with open("high_freq_meta_cam_1.pkl", "wb") as f:
    #     pickle.dump(high_freq_meta_cam_1, f)
    # with open("high_freq_meta_cam_2.pkl", "wb") as f:
    #     pickle.dump(high_freq_meta_cam_2, f)

    with open("low_freq_cka_meta_cam_1.pkl", "wb") as f:
        pickle.dump(low_freq_cka_meta_cam_1, f)
    with open("low_freq_cka_meta_cam_2.pkl", "wb") as f:
        pickle.dump(low_freq_cka_meta_cam_2, f)
    with open("high_freq_cka_meta_cam_1.pkl", "wb") as f:
        pickle.dump(high_freq_cka_meta_cam_1, f)
    with open("high_freq_cka_meta_cam_2.pkl", "wb") as f:
        pickle.dump(high_freq_cka_meta_cam_2, f)

    # with open("low_freq_sem_meta_cam_1.pkl", "wb") as f:
    #     pickle.dump(low_freq_sem_meta_cam_1, f)
    # with open("low_freq_sem_meta_cam_2.pkl", "wb") as f:
    #     pickle.dump(low_freq_sem_meta_cam_2, f)
    # with open("high_freq_sem_meta_cam_1.pkl", "wb") as f:
    #     pickle.dump(high_freq_sem_meta_cam_1, f)
    # with open("high_freq_sem_meta_cam_2.pkl", "wb") as f:
    #     pickle.dump(high_freq_sem_meta_cam_2, f)

if __name__ == "__main__":
    run()
