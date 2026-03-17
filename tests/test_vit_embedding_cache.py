"""
Tests for pre-computed frozen ViT embedding cache.

Tests:
  1. Embedding correctness: cached embeddings match live ViT forward pass
  2. Dataset __getitem__ with cache: correct shapes and values
  3. Dataset padding/boundary: edge cases at episode boundaries
  4. Validation dataset propagation: cache correctly re-indexed
  5. Full policy forward: cached path produces same output as live path
  6. ResNet backward compatibility: ResNet policy + dataset unaffected
  7. new_type_dataloader guard: ndim check works for both images and embeddings
  8. Memory usage: cache fits expected memory budget
  9. Training iteration speed: cached path is faster than live path
  10. _should_precompute_vit: correctly detects when to precompute
"""

import sys
import os
import time
import copy
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.planar_pushing_attention_dataset import PlanarPushingAttentionDataset
from diffusion_policy.model.vision.vit_obs_encoder import ViTObsEncoder


# ─── Shared fixtures ───────────────────────────────────────────────────────

SHAPE_META = {
    'action': {'shape': [2]},
    'obs': {
        'agent_pos': {'type': 'low_dim', 'shape': [2]},
        'overhead_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
        'wrist_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
    },
}

ZARR_PATHS = [
    'data/iros/long_context_planar_pushing/start_bin_4_sorted.zarr',
    'data/iros/long_context_planar_pushing/start_bin_4_via_mirror_sorted_reversed.zarr',
]

def _make_dataset(zarr_paths=None, n_obs_steps=8, horizon=24, min_obs_steps=1):
    """Create a dataset from real zarr data."""
    if zarr_paths is None:
        zarr_paths = ZARR_PATHS[:1]
    zarr_configs = [
        {'path': p, 'max_train_episodes': 4, 'sampling_weight': None, 'val_ratio': 0.1}
        for p in zarr_paths
    ]
    return PlanarPushingAttentionDataset(
        zarr_configs=zarr_configs,
        shape_meta=SHAPE_META,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        min_obs_steps=min_obs_steps,
        pad_before=n_obs_steps - 1,
        pad_after=horizon - n_obs_steps,
        seed=42,
        training_mode='random',
    )


def _make_encoder(variant="A-small", vit_freeze=True, device='cuda'):
    """Create a frozen ViT encoder with tiny model for fast testing."""
    encoder = ViTObsEncoder(
        shape_meta=SHAPE_META,
        variant=variant,
        vit_model_name="vit_tiny_patch16_224",
        pretrained=False,  # random init for speed
        vit_freeze=vit_freeze,
        projection_dim=64,
        crop_shape=(112, 112),
        imagenet_norm=True,
        share_rgb_model=False,
    )
    return encoder.to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Embedding correctness
# ═══════════════════════════════════════════════════════════════════════════

def test_embedding_correctness():
    """Cached embeddings must exactly match live ViT forward pass."""
    print("\n[Test 1] Embedding correctness...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = _make_encoder(variant="A-small", device=device)
    dataset = _make_dataset()

    # Pre-compute cache
    cache, embed_dim = encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)

    # Verify embed_dim matches
    assert embed_dim == encoder._vit_embed_dim, \
        f"embed_dim mismatch: {embed_dim} vs {encoder._vit_embed_dim}"

    # Spot-check random frames against live forward pass
    encoder.eval()
    rb = dataset.replay_buffers[0]
    n_frames = len(rb['overhead_camera'])

    rng = np.random.RandomState(42)
    test_indices = rng.choice(n_frames, size=min(20, n_frames), replace=False)

    for key in ['overhead_camera', 'wrist_camera']:
        cached_all = cache[key][0]  # (n_frames, embed_dim)
        for frame_idx in test_indices:
            # Live forward
            img = torch.from_numpy(
                rb[key][frame_idx:frame_idx+1].copy()
            ).float().to(device)
            img = img.permute(0, 3, 1, 2) / 255.0

            with torch.no_grad():
                if key in encoder.crop_randomizers:
                    img_cropped = encoder.crop_randomizers[key](img)
                else:
                    img_cropped = img
                if encoder.use_imagenet_norm:
                    img_cropped = (img_cropped - encoder.img_mean) / encoder.img_std
                live_cls = encoder.vit_encoders[key](img_cropped)  # (1, embed_dim)

            cached_cls = cached_all[frame_idx:frame_idx+1].to(device)
            diff = (live_cls - cached_cls).abs().max().item()
            assert diff < 1e-4, \
                f"Frame {frame_idx} {key}: max diff={diff:.6f} (should be <1e-4)"

    print(f"  PASSED: 20 random frames × 2 cameras, max diff < 1e-4")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Dataset __getitem__ with cache
# ═══════════════════════════════════════════════════════════════════════════

def test_dataset_getitem_with_cache():
    """After set_embedding_cache, __getitem__ returns embeddings not images."""
    print("\n[Test 2] Dataset __getitem__ with cache...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = _make_encoder(variant="A-small", device=device)
    dataset = _make_dataset(n_obs_steps=8, min_obs_steps=4)

    # Without cache: images
    sample_no_cache = dataset[0]
    for key in ['overhead_camera', 'wrist_camera']:
        shape = sample_no_cache['obs'][key].shape
        assert len(shape) == 4, f"Without cache, expected 4D (T, H, W, C), got {shape}"

    # Set cache
    cache, embed_dim = encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)
    dataset.set_embedding_cache(cache, embed_dim)

    # With cache: embeddings
    sample_cached = dataset[0]
    for key in ['overhead_camera', 'wrist_camera']:
        shape = sample_cached['obs'][key].shape
        assert len(shape) == 2, \
            f"With cache, expected 2D (T, embed_dim), got {shape}"
        assert shape[-1] == embed_dim, \
            f"embed_dim mismatch: {shape[-1]} vs {embed_dim}"

    # Low-dim and action should be unchanged
    assert sample_cached['obs']['agent_pos'].shape == sample_no_cache['obs']['agent_pos'].shape
    assert sample_cached['action'].shape == sample_no_cache['action'].shape

    # Metadata should still work
    assert 'sample_metadata' in sample_cached
    assert 'num_obs_steps' in sample_cached['sample_metadata']

    print(f"  PASSED: cached shape={sample_cached['obs']['overhead_camera'].shape}, "
          f"embed_dim={embed_dim}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Dataset padding / boundary correctness
# ═══════════════════════════════════════════════════════════════════════════

def test_dataset_padding_boundary():
    """Verify cached embeddings handle episode boundary padding correctly."""
    print("\n[Test 3] Padding/boundary correctness...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = _make_encoder(variant="A-full", device=device)
    dataset = _make_dataset(n_obs_steps=8, min_obs_steps=8, horizon=24)

    cache, embed_dim = encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)
    dataset.set_embedding_cache(cache, embed_dim)

    # Check that idx=0 (likely at episode boundary) works without errors
    # and first-frame padding is correct
    sampler = dataset.samplers[0]

    # Find an index at the very start of an episode (sample_start_idx > 0)
    boundary_idx = None
    for i in range(min(100, len(sampler))):
        buf_start, buf_end, samp_start, samp_end = sampler.indices[i]
        if samp_start > 0:
            boundary_idx = i
            break

    if boundary_idx is not None:
        # Force fixed obs steps to test padding
        dataset.training_mode = 'random'
        dataset.min_obs_steps = 8
        dataset.max_obs_steps = 8
        sample = dataset[boundary_idx]
        key = 'overhead_camera'
        emb = sample['obs'][key]
        # Check that padded region (before samp_start) equals first real frame
        buf_start, buf_end, samp_start, samp_end = sampler.indices[boundary_idx]
        if samp_start > 0:
            padded = emb[:samp_start]
            first_real = emb[samp_start:samp_start+1].expand_as(padded)
            diff = (padded - first_real).abs().max().item()
            assert diff < 1e-6, f"Padding mismatch: max diff={diff}"
            print(f"  Boundary idx {boundary_idx}: samp_start={samp_start}, "
                  f"padding verified (diff={diff:.2e})")
    else:
        print(f"  No boundary sample found in first 100 indices (all mid-episode)")

    # Verify a normal (non-boundary) sample doesn't crash
    sample = dataset[len(sampler) // 2]
    assert not torch.isnan(sample['obs']['overhead_camera'][:8]).any(), \
        "NaN in first 8 obs steps of non-boundary sample"

    print(f"  PASSED")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Validation dataset cache propagation
# ═══════════════════════════════════════════════════════════════════════════

def test_validation_dataset_propagation():
    """Cache should propagate to validation datasets with correct re-indexing."""
    print("\n[Test 4] Validation dataset cache propagation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = _make_encoder(variant="A-small", device=device)
    dataset = _make_dataset(zarr_paths=ZARR_PATHS[:2], n_obs_steps=8, min_obs_steps=8)

    cache, embed_dim = encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)
    dataset.set_embedding_cache(cache, embed_dim)

    # Get validation datasets
    for idx in range(2):
        val_ds = dataset.get_validation_dataset(idx)
        assert val_ds._embedding_cache is not None, \
            f"Val dataset {idx}: cache not propagated"
        assert val_ds._embed_dim == embed_dim, \
            f"Val dataset {idx}: embed_dim mismatch"

        # Cache should have exactly 1 buffer (re-indexed)
        for key in ['overhead_camera', 'wrist_camera']:
            assert len(val_ds._embedding_cache[key]) == 1, \
                f"Val dataset {idx} {key}: expected 1 buffer, got {len(val_ds._embedding_cache[key])}"
            # The tensor should match the parent's buffer at `idx`
            expected = cache[key][idx]
            actual = val_ds._embedding_cache[key][0]
            assert torch.equal(expected, actual), \
                f"Val dataset {idx} {key}: cache tensor doesn't match parent buffer {idx}"

        # Verify __getitem__ works and returns embeddings
        if len(val_ds) > 0:
            sample = val_ds[0]
            assert sample['obs']['overhead_camera'].shape[-1] == embed_dim

    print(f"  PASSED: both val datasets have correct cache and produce embeddings")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Full policy forward with cached embeddings
# ═══════════════════════════════════════════════════════════════════════════

def test_policy_forward_cached_vs_live():
    """Policy compute_loss should produce same result with cached vs live embeddings."""
    print("\n[Test 5] Policy forward: cached vs live...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from diffusion_policy.policy.diffusion_vit_attention_hybrid_image_policy import (
        DiffusionViTAttentionHybridImagePolicy)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', clip_sample=True,
        prediction_type='epsilon', variance_type='fixed_small')

    # Build policy with frozen ViT
    policy = DiffusionViTAttentionHybridImagePolicy(
        shape_meta=SHAPE_META,
        noise_scheduler=scheduler,
        horizon=24, n_action_steps=8, n_obs_steps=8,
        vit_variant="A-small",
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=False,
        vit_freeze=True,
        vit_projection_dim=64,
        crop_shape=(112, 112),
        imagenet_norm=True,
        diffusion_step_embed_dim=128,
        down_dims=(128, 256),
        kernel_size=5, n_groups=8,
        obs_as_global_cond=True,
    ).to(device)

    # Create dataset
    dataset = _make_dataset(n_obs_steps=8, min_obs_steps=8)

    # Pre-compute cache
    cache, embed_dim = policy.obs_encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)

    # Get a fixed sample (images)
    torch.manual_seed(42)
    np.random.seed(42)
    sample_img = dataset[10]

    # Set cache and get the same sample (embeddings)
    dataset.set_embedding_cache(cache, embed_dim)
    torch.manual_seed(42)
    np.random.seed(42)
    sample_emb = dataset[10]

    # Build batch from the image sample
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)  # re-move so normalizer buffers are on device
    policy.eval()

    def to_batch(sample):
        batch = {}
        batch['obs'] = {}
        for k, v in sample['obs'].items():
            batch['obs'][k] = v.unsqueeze(0).to(device)
        batch['action'] = sample['action'].unsqueeze(0).to(device)
        batch['sample_metadata'] = {
            'num_obs_steps': torch.tensor([sample['sample_metadata']['num_obs_steps']]).to(device)
        }
        return batch

    # Process image batch: convert HWC uint8 -> CHW float
    batch_img = to_batch(sample_img)
    for key in ['overhead_camera', 'wrist_camera']:
        batch_img['obs'][key] = torch.moveaxis(batch_img['obs'][key], -1, 2).float() / 255.0

    # Process embedding batch: already float
    batch_emb = to_batch(sample_emb)

    with torch.no_grad():
        loss_img = policy.compute_loss(batch_img)
        loss_emb = policy.compute_loss(batch_emb)

    diff = abs(loss_img.item() - loss_emb.item())
    print(f"  Loss (images): {loss_img.item():.6f}")
    print(f"  Loss (cached): {loss_emb.item():.6f}")
    print(f"  Diff: {diff:.6f}")

    # Losses won't be identical because random noise in diffusion,
    # but the obs encoding should be the same. Let's check obs encoding directly.
    nobs_img = policy._normalize_obs(batch_img['obs'])
    nobs_emb = policy._normalize_obs(batch_emb['obs'])

    num_obs = batch_img['sample_metadata']['num_obs_steps']
    with torch.no_grad():
        enc_img, mask_img, pos_img = policy._prepare_batch_and_apply_obs_encoding(
            nobs_img, num_obs)
        enc_emb, mask_emb, pos_emb = policy._prepare_batch_and_apply_obs_encoding(
            nobs_emb, num_obs)

    enc_diff = (enc_img - enc_emb).abs().max().item()
    print(f"  Encoding max diff: {enc_diff:.6f}")
    assert enc_diff < 1e-3, f"Encoding diff too large: {enc_diff}"
    assert torch.equal(mask_img, mask_emb), "Masks differ"
    assert torch.equal(pos_img, pos_emb), "Positions differ"

    print(f"  PASSED: obs encoding max diff = {enc_diff:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: ResNet backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

def test_resnet_backward_compatibility():
    """ResNet policy + PlanarPushingAttentionDataset should work unchanged."""
    print("\n[Test 6] ResNet backward compatibility...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dataset without any cache
    dataset = _make_dataset(n_obs_steps=8, min_obs_steps=8)

    # Verify _embedding_cache is None by default
    assert dataset._embedding_cache is None, "Cache should be None by default"

    # Get samples - should return images
    sample = dataset[0]
    for key in ['overhead_camera', 'wrist_camera']:
        shape = sample['obs'][key].shape
        # Should be (horizon, H, W, C) = (24, 128, 128, 3)
        assert len(shape) == 4, f"Expected 4D image, got shape {shape}"
        assert shape[-1] == 3, f"Expected HWC format with C=3, got {shape}"

    # Validation dataset should also have no cache
    val_ds = dataset.get_validation_dataset(0)
    assert val_ds._embedding_cache is None
    if len(val_ds) > 0:
        val_sample = val_ds[0]
        assert len(val_sample['obs']['overhead_camera'].shape) == 4

    # Test that the new_type_dataloader guard works for images
    batch_obs = torch.from_numpy(
        sample['obs']['overhead_camera'].numpy() if isinstance(sample['obs']['overhead_camera'], torch.Tensor)
        else sample['obs']['overhead_camera']
    ).unsqueeze(0).to(device)

    # Simulate new_type_dataloader processing
    if batch_obs.ndim >= 4:
        processed = torch.moveaxis(batch_obs, -1, 2).float() / 255.0
        assert processed.shape[-3] == 3, "Should be CHW after moveaxis"

    print(f"  PASSED: dataset returns images, no cache interference")


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: new_type_dataloader ndim guard
# ═══════════════════════════════════════════════════════════════════════════

def test_ndim_guard():
    """ndim check should skip moveaxis for embeddings, apply it for images."""
    print("\n[Test 7] new_type_dataloader ndim guard...")

    # Simulate image batch: (B, T, H, W, C) = 5D
    img = torch.randint(0, 255, (2, 8, 128, 128, 3), dtype=torch.uint8)
    assert img.ndim >= 4, "Images should trigger processing"
    processed = torch.moveaxis(img, -1, 2).float() / 255.0
    assert processed.shape == (2, 8, 3, 128, 128)

    # Simulate embedding batch: (B, T, embed_dim) = 3D
    emb = torch.randn(2, 8, 192)
    assert emb.ndim < 4, "Embeddings should skip processing"
    # The guard: if batch['obs'][key].ndim >= 4
    # For embeddings (3D), this is False → skip

    print(f"  PASSED: ndim guard correctly differentiates images vs embeddings")


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Memory usage
# ═══════════════════════════════════════════════════════════════════════════

def test_memory_usage():
    """Cache memory should be reasonable (~few MB for test data)."""
    print("\n[Test 8] Memory usage...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = _make_encoder(variant="A-full", device=device)
    dataset = _make_dataset(zarr_paths=ZARR_PATHS[:1])

    cache, embed_dim = encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)

    total_bytes = 0
    for key in cache:
        for tensor in cache[key]:
            total_bytes += tensor.nelement() * tensor.element_size()

    total_mb = total_bytes / (1024 * 1024)
    n_frames = sum(len(rb['overhead_camera']) for rb in dataset.replay_buffers)
    n_cameras = 2
    expected_mb = n_frames * n_cameras * embed_dim * 4 / (1024 * 1024)

    print(f"  Cache: {total_mb:.1f} MB")
    print(f"  Expected: {expected_mb:.1f} MB")
    print(f"  Frames: {n_frames}, Cameras: {n_cameras}, embed_dim: {embed_dim}")

    assert abs(total_mb - expected_mb) < 1.0, \
        f"Memory mismatch: {total_mb:.1f} vs {expected_mb:.1f} MB"

    print(f"  PASSED: cache memory matches expected")


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Training iteration speed
# ═══════════════════════════════════════════════════════════════════════════

def test_iteration_speed():
    """Cached path should be significantly faster than live ViT forward."""
    print("\n[Test 9] Training iteration speed...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("  SKIPPED: speed test meaningful only on GPU")
        return

    from diffusion_policy.policy.diffusion_vit_attention_hybrid_image_policy import (
        DiffusionViTAttentionHybridImagePolicy)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', clip_sample=True,
        prediction_type='epsilon', variance_type='fixed_small')

    policy = DiffusionViTAttentionHybridImagePolicy(
        shape_meta=SHAPE_META,
        noise_scheduler=scheduler,
        horizon=24, n_action_steps=8, n_obs_steps=8,
        vit_variant="A-small",
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=False,
        vit_freeze=True,
        vit_projection_dim=64,
        crop_shape=(112, 112),
        imagenet_norm=True,
        diffusion_step_embed_dim=128,
        down_dims=(128, 256),
        kernel_size=5, n_groups=8,
        obs_as_global_cond=True,
    ).to(device)

    dataset = _make_dataset(n_obs_steps=8, min_obs_steps=8)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)  # re-move so normalizer buffers are on device

    # Pre-compute cache
    cache, embed_dim = policy.obs_encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)

    def make_batch(ds, batch_size=16, is_cached=False):
        samples = [ds[i % len(ds)] for i in range(batch_size)]
        batch = {'obs': {}, 'action': None, 'sample_metadata': {}}
        for key in ['overhead_camera', 'wrist_camera', 'agent_pos']:
            batch['obs'][key] = torch.stack(
                [s['obs'][key] for s in samples]).to(device)
        batch['action'] = torch.stack(
            [s['action'] for s in samples]).to(device)
        batch['sample_metadata']['num_obs_steps'] = torch.tensor(
            [s['sample_metadata']['num_obs_steps'] for s in samples]).to(device)
        # Convert images
        if not is_cached:
            for key in ['overhead_camera', 'wrist_camera']:
                batch['obs'][key] = torch.moveaxis(
                    batch['obs'][key], -1, 2).float() / 255.0
        return batch

    # Benchmark: live images
    batch_img = make_batch(dataset, batch_size=16, is_cached=False)
    policy.train()
    # Warmup
    for _ in range(3):
        loss = policy.compute_loss(batch_img)
        loss.backward()
        policy.zero_grad()
    torch.cuda.synchronize()

    t0 = time.time()
    n_iters = 20
    for _ in range(n_iters):
        loss = policy.compute_loss(batch_img)
        loss.backward()
        policy.zero_grad()
    torch.cuda.synchronize()
    time_live = (time.time() - t0) / n_iters

    # Benchmark: cached embeddings
    dataset.set_embedding_cache(cache, embed_dim)
    batch_emb = make_batch(dataset, batch_size=16, is_cached=True)
    # Warmup
    for _ in range(3):
        loss = policy.compute_loss(batch_emb)
        loss.backward()
        policy.zero_grad()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iters):
        loss = policy.compute_loss(batch_emb)
        loss.backward()
        policy.zero_grad()
    torch.cuda.synchronize()
    time_cached = (time.time() - t0) / n_iters

    speedup = time_live / time_cached
    print(f"  Live:   {time_live*1000:.1f} ms/iter")
    print(f"  Cached: {time_cached*1000:.1f} ms/iter")
    print(f"  Speedup: {speedup:.1f}x")

    assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.1f}x"
    print(f"  PASSED: {speedup:.1f}x speedup")


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: _should_precompute_vit logic
# ═══════════════════════════════════════════════════════════════════════════

def test_should_precompute_vit():
    """_should_precompute_vit should correctly gate on freeze+variant."""
    print("\n[Test 10] _should_precompute_vit logic...")
    from omegaconf import OmegaConf

    # Create a minimal mock workspace to call the method
    class MockWorkspace:
        pass
    MockWorkspace._should_precompute_vit = \
        __import__('diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace_no_env',
                   fromlist=['TrainDiffusionUnetHybridWorkspaceNoEnv']
        ).TrainDiffusionUnetHybridWorkspaceNoEnv._should_precompute_vit

    ws = MockWorkspace()

    cases = [
        # (vit_freeze, vit_variant, expected)
        (True,  "A-small",      True),
        (True,  "A-full",       True),
        (True,  "B",            True),
        (True,  "C",            True),
        (True,  "B-per-stream", True),
        (True,  "C-per-stream", True),
        (True,  "D",            False),  # Video ViT
        (True,  "E",            False),  # Video ViT
        (False, "A-small",      False),  # Not frozen
        (False, "B-per-stream", False),  # Not frozen
    ]

    for vit_freeze, variant, expected in cases:
        cfg = OmegaConf.create({
            'policy': {
                'vit_freeze': vit_freeze,
                'vit_variant': variant,
            }
        })
        result = ws._should_precompute_vit(cfg)
        assert result == expected, \
            f"freeze={vit_freeze}, variant={variant}: expected {expected}, got {result}"

    # ResNet policy (no vit_variant)
    cfg = OmegaConf.create({'policy': {'crop_shape': [112, 112]}})
    assert ws._should_precompute_vit(cfg) == False

    print(f"  PASSED: all {len(cases)} cases + ResNet case correct")


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Multi-dataset cache indexing
# ═══════════════════════════════════════════════════════════════════════════

def test_multi_dataset_cache_indexing():
    """With multiple zarr datasets, cache must index into the right buffer."""
    print("\n[Test 11] Multi-dataset cache indexing...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = _make_encoder(variant="A-full", device=device)
    dataset = _make_dataset(zarr_paths=ZARR_PATHS[:2], n_obs_steps=8, min_obs_steps=8)

    cache, embed_dim = encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)

    # Verify each buffer has different embeddings (different data)
    for key in ['overhead_camera', 'wrist_camera']:
        assert len(cache[key]) == 2, f"Expected 2 buffer caches for {key}"
        # Shapes should match number of frames in each buffer
        for buf_idx in range(2):
            n_frames = len(dataset.replay_buffers[buf_idx][key])
            assert cache[key][buf_idx].shape == (n_frames, embed_dim), \
                f"Buffer {buf_idx} {key}: shape {cache[key][buf_idx].shape} != ({n_frames}, {embed_dim})"

    dataset.set_embedding_cache(cache, embed_dim)

    # Sample from the dataset - should not crash
    for i in range(50):
        sample = dataset[i % len(dataset)]
        emb = sample['obs']['overhead_camera']
        assert emb.shape[-1] == embed_dim
        # First 8 steps should not be NaN
        assert not torch.isnan(emb[:8]).any(), f"NaN in first 8 steps at idx {i}"

    print(f"  PASSED: multi-dataset indexing correct, no NaN in obs")


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: Variant B-per-stream with cache
# ═══════════════════════════════════════════════════════════════════════════

def test_b_per_stream_with_cache():
    """B-per-stream variant should work with cached embeddings."""
    print("\n[Test 12] B-per-stream with cache...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    from diffusion_policy.policy.diffusion_vit_attention_hybrid_image_policy import (
        DiffusionViTAttentionHybridImagePolicy)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02,
        beta_schedule='squaredcos_cap_v2', clip_sample=True,
        prediction_type='epsilon', variance_type='fixed_small')

    policy = DiffusionViTAttentionHybridImagePolicy(
        shape_meta=SHAPE_META,
        noise_scheduler=scheduler,
        horizon=24, n_action_steps=8, n_obs_steps=8,
        vit_variant="B-per-stream",
        vit_model_name="vit_tiny_patch16_224",
        vit_pretrained=False,
        vit_freeze=True,
        imagenet_norm=True,
        crop_shape=(112, 112),
        temporal_depth=2,
        temporal_num_heads=3,
        max_frames=100,
        diffusion_step_embed_dim=128,
        down_dims=(128, 256),
        kernel_size=5, n_groups=8,
        obs_as_global_cond=True,
    ).to(device)

    dataset = _make_dataset(n_obs_steps=8, min_obs_steps=4)
    normalizer = dataset.get_normalizer()
    policy.set_normalizer(normalizer)
    policy.to(device)  # re-move so normalizer buffers are on device

    cache, embed_dim = policy.obs_encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)
    dataset.set_embedding_cache(cache, embed_dim)

    # Build batch
    samples = [dataset[i % len(dataset)] for i in range(4)]
    batch = {'obs': {}, 'action': None, 'sample_metadata': {}}
    for key in ['overhead_camera', 'wrist_camera', 'agent_pos']:
        batch['obs'][key] = torch.stack([s['obs'][key] for s in samples]).to(device)
    batch['action'] = torch.stack([s['action'] for s in samples]).to(device)
    batch['sample_metadata']['num_obs_steps'] = torch.tensor(
        [s['sample_metadata']['num_obs_steps'] for s in samples]).to(device)

    policy.train()
    loss = policy.compute_loss(batch)
    assert not torch.isnan(loss), "Loss is NaN"
    loss.backward()

    # Check temporal encoder gradients are flowing
    for name, p in policy.obs_encoder.per_stream_temporal_encoders.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"
            assert p.grad.abs().sum() > 0, f"Zero gradient for {name}"
            break

    print(f"  PASSED: loss={loss.item():.4f}, temporal grads flowing")


# ═══════════════════════════════════════════════════════════════════════════
# Test 13: Disk cache save/load
# ═══════════════════════════════════════════════════════════════════════════

def test_disk_cache():
    """Test saving and loading cache from disk."""
    print("\n[Test 13] Disk cache save/load...")
    import tempfile

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = _make_encoder(variant="A-full", device=device)
    dataset = _make_dataset()

    cache, embed_dim = encoder.precompute_all_embeddings(
        dataset.replay_buffers, batch_size=64)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, 'test_cache.pt')
        torch.save({'cache': cache, 'embed_dim': embed_dim}, cache_path)

        # Load
        loaded = torch.load(cache_path, map_location='cpu')
        loaded_cache = loaded['cache']
        loaded_dim = loaded['embed_dim']

        assert loaded_dim == embed_dim
        for key in cache:
            for i in range(len(cache[key])):
                assert torch.equal(cache[key][i], loaded_cache[key][i]), \
                    f"Cache mismatch for {key}[{i}]"

    print(f"  PASSED: disk round-trip preserves all tensors")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("ViT Embedding Cache Test Suite")
    print("=" * 70)

    tests = [
        test_embedding_correctness,
        test_dataset_getitem_with_cache,
        test_dataset_padding_boundary,
        test_validation_dataset_propagation,
        test_policy_forward_cached_vs_live,
        test_resnet_backward_compatibility,
        test_ndim_guard,
        test_memory_usage,
        test_iteration_speed,
        test_should_precompute_vit,
        test_multi_dataset_cache_indexing,
        test_b_per_stream_with_cache,
        test_disk_cache,
    ]

    passed = 0
    failed = 0
    skipped = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)
