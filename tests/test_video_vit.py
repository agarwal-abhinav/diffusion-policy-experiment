"""
Test Video ViT encoder variants D and E.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn


def test_video_vit_encoder_cls_only():
    """Test VideoViTEncoder with cls_only (variant D) temporal mode."""
    from diffusion_policy.model.vision.vit_obs_encoder import VideoViTEncoder
    import timm

    vit = timm.create_model('vit_small_patch16_224', pretrained=False,
                            img_size=112, num_classes=0)
    encoder = VideoViTEncoder(
        vit, temporal_mode="cls_only", temporal_every_k=3,
        max_frames=100, temporal_num_heads=6)

    B, T, C, H, W = 2, 5, 3, 112, 112
    x = torch.randn(B * T, C, H, W)

    with torch.no_grad():
        out = encoder(x, T)

    assert out.shape == (B * T, 384), f"Expected ({B*T}, 384), got {out.shape}"
    print(f"  cls_only forward: OK, output shape {out.shape}")

    # Test with padding mask
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    padding_mask[1, 3:] = True  # sample 1 only has 3 valid frames

    with torch.no_grad():
        out_masked = encoder(x, T, key_padding_mask=padding_mask)

    assert out_masked.shape == (B * T, 384)
    print(f"  cls_only with padding mask: OK")


def test_video_vit_encoder_divided_st():
    """Test VideoViTEncoder with divided_st (variant E) temporal mode."""
    from diffusion_policy.model.vision.vit_obs_encoder import VideoViTEncoder
    import timm

    vit = timm.create_model('vit_small_patch16_224', pretrained=False,
                            img_size=112, num_classes=0)
    encoder = VideoViTEncoder(
        vit, temporal_mode="divided_st", temporal_every_k=4,
        max_frames=100, temporal_num_heads=6)

    B, T, C, H, W = 2, 4, 3, 112, 112
    x = torch.randn(B * T, C, H, W)

    with torch.no_grad():
        out = encoder(x, T)

    assert out.shape == (B * T, 384), f"Expected ({B*T}, 384), got {out.shape}"
    print(f"  divided_st forward: OK, output shape {out.shape}")

    # Test with padding mask
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    padding_mask[0, 2:] = True

    with torch.no_grad():
        out_masked = encoder(x, T, key_padding_mask=padding_mask)

    assert out_masked.shape == (B * T, 384)
    print(f"  divided_st with padding mask: OK")


def test_vit_obs_encoder_variant_d():
    """Test ViTObsEncoder with variant D (CLS mid-fusion)."""
    from diffusion_policy.model.vision.vit_obs_encoder import ViTObsEncoder

    shape_meta = {
        'obs': {
            'overhead_camera': {'shape': (3, 128, 128), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)},
    }

    encoder = ViTObsEncoder(
        shape_meta=shape_meta,
        variant="D",
        vit_model_name="vit_small_patch16_224",
        pretrained=False,
        crop_shape=(112, 112),
        temporal_every_k=3,
        temporal_num_heads=6,
        max_frames=100,
    )

    assert encoder.is_video_vit
    assert not encoder.needs_temporal
    print(f"  variant D: output_dim={encoder.output_dim}")

    # Test forward_video
    B, T = 2, 5
    obs_dict = {
        'overhead_camera': torch.randn(B * T, 3, 128, 128),
        'agent_pos': torch.randn(B * T, 2),
    }
    padding_mask = torch.zeros(B, T, dtype=torch.bool)
    padding_mask[1, 3:] = True

    with torch.no_grad():
        out = encoder.forward_video(obs_dict, T, key_padding_mask=padding_mask)

    assert out.shape == (B * T, encoder.output_dim)
    print(f"  variant D forward_video: OK, output shape {out.shape}")


def test_vit_obs_encoder_variant_e():
    """Test ViTObsEncoder with variant E (divided space-time)."""
    from diffusion_policy.model.vision.vit_obs_encoder import ViTObsEncoder

    shape_meta = {
        'obs': {
            'overhead_camera': {'shape': (3, 128, 128), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)},
    }

    encoder = ViTObsEncoder(
        shape_meta=shape_meta,
        variant="E",
        vit_model_name="vit_small_patch16_224",
        pretrained=False,
        crop_shape=(112, 112),
        temporal_every_k=4,
        temporal_num_heads=6,
        max_frames=100,
    )

    assert encoder.is_video_vit
    print(f"  variant E: output_dim={encoder.output_dim}")

    B, T = 2, 4
    obs_dict = {
        'overhead_camera': torch.randn(B * T, 3, 128, 128),
        'agent_pos': torch.randn(B * T, 2),
    }

    with torch.no_grad():
        out = encoder.forward_video(obs_dict, T)

    assert out.shape == (B * T, encoder.output_dim)
    print(f"  variant E forward_video: OK, output shape {out.shape}")


def test_existing_variants_unbroken():
    """Verify A-small still works after D/E additions."""
    from diffusion_policy.model.vision.vit_obs_encoder import ViTObsEncoder

    shape_meta = {
        'obs': {
            'overhead_camera': {'shape': (3, 128, 128), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)},
    }

    encoder = ViTObsEncoder(
        shape_meta=shape_meta,
        variant="A-small",
        vit_model_name="vit_small_patch16_224",
        pretrained=False,
        crop_shape=(112, 112),
    )

    assert not encoder.is_video_vit
    assert not encoder.needs_temporal

    N = 10
    obs_dict = {
        'overhead_camera': torch.randn(N, 3, 128, 128),
        'agent_pos': torch.randn(N, 2),
    }

    with torch.no_grad():
        out = encoder(obs_dict)

    assert out.shape == (N, encoder.output_dim)
    print(f"  A-small backward compat: OK, output shape {out.shape}")


def test_temporal_attention_affects_output():
    """Verify that temporal attention actually changes outputs vs independent frames."""
    from diffusion_policy.model.vision.vit_obs_encoder import VideoViTEncoder
    import timm

    torch.manual_seed(42)
    vit = timm.create_model('vit_small_patch16_224', pretrained=False,
                            img_size=112, num_classes=0)
    encoder = VideoViTEncoder(
        vit, temporal_mode="cls_only", temporal_every_k=3,
        max_frames=100, temporal_num_heads=6)
    encoder.eval()

    B, T, C, H, W = 1, 4, 3, 112, 112
    x = torch.randn(B * T, C, H, W)

    with torch.no_grad():
        out_video = encoder(x, T)
        # Compare: process each frame independently (T=1)
        out_indep = torch.cat([encoder(x[i:i+1], 1) for i in range(T)])

    # Outputs should differ because temporal attention mixes info across frames
    diff = (out_video - out_indep).abs().mean().item()
    assert diff > 1e-5, f"Temporal attention had no effect! diff={diff}"
    print(f"  Temporal attention effect: mean diff = {diff:.6f}")


def test_policy_with_variant_d():
    """End-to-end test: policy compute_loss and predict_action with variant D."""
    from diffusion_policy.policy.diffusion_vit_attention_hybrid_image_policy import (
        DiffusionViTAttentionHybridImagePolicy)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    shape_meta = {
        'obs': {
            'overhead_camera': {'shape': (3, 128, 128), 'type': 'rgb'},
            'agent_pos': {'shape': (2,), 'type': 'low_dim'},
        },
        'action': {'shape': (2,)},
    }

    scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        variance_type='fixed_small',
        clip_sample=True,
        prediction_type='epsilon',
    )

    policy = DiffusionViTAttentionHybridImagePolicy(
        shape_meta=shape_meta,
        noise_scheduler=scheduler,
        horizon=96,
        n_action_steps=8,
        n_obs_steps=80,
        vit_variant="D",
        vit_model_name="vit_small_patch16_224",
        vit_pretrained=False,
        crop_shape=(112, 112),
        temporal_every_k=4,
        temporal_num_heads=6,
        max_frames=100,
        down_dims=(128, 256),
        n_past_action_steps=8,
    )

    # Setup normalizer — fit each key separately
    normalizer = LinearNormalizer()
    normalizer.fit(
        data={
            'action': torch.randn(100, 2),
            'overhead_camera': torch.rand(100, 3, 128, 128),
            'agent_pos': torch.randn(100, 2),
        },
        mode='limits', output_max=1.0, output_min=-1.0)
    policy.set_normalizer(normalizer)

    # Test compute_loss
    B = 2
    batch = {
        'obs': {
            'overhead_camera': torch.rand(B, 80, 3, 128, 128),
            'agent_pos': torch.randn(B, 80, 2),
        },
        'action': torch.randn(B, 96, 2),
        'sample_metadata': {
            'num_obs_steps': torch.tensor([80, 40]),
        },
    }

    loss = policy.compute_loss(batch)
    assert loss.ndim == 0 and loss.item() > 0
    print(f"  compute_loss: OK, loss={loss.item():.4f}")

    # Test predict_action
    obs_dict = {
        'obs': {
            'overhead_camera': torch.rand(1, 80, 3, 128, 128),
            'agent_pos': torch.randn(1, 80, 2),
        },
    }

    with torch.no_grad():
        result = policy.predict_action(obs_dict, use_DDIM=True)

    assert result['action'].shape == (1, 8, 2)
    print(f"  predict_action: OK, shape={result['action'].shape}")


if __name__ == '__main__':
    print("=== VideoViTEncoder CLS-only (D) ===")
    test_video_vit_encoder_cls_only()
    print()
    print("=== VideoViTEncoder Divided Space-Time (E) ===")
    test_video_vit_encoder_divided_st()
    print()
    print("=== ViTObsEncoder Variant D ===")
    test_vit_obs_encoder_variant_d()
    print()
    print("=== ViTObsEncoder Variant E ===")
    test_vit_obs_encoder_variant_e()
    print()
    print("=== Existing Variants Unbroken ===")
    test_existing_variants_unbroken()
    print()
    print("=== Temporal Attention Effect ===")
    test_temporal_attention_affects_output()
    print()
    print("=== Policy with Variant D (end-to-end) ===")
    test_policy_with_variant_d()
    print()
    print("ALL VIDEO VIT TESTS PASSED!")
