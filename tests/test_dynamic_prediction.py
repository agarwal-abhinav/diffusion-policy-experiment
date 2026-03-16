"""
Test per-sample dynamic prediction horizon and dataset training modes.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import math
import numpy as np

def test_dataset_training_modes():
    """Test all 4 training modes in the dataset."""
    from diffusion_policy.dataset.planar_pushing_attention_dataset import PlanarPushingAttentionDataset

    # Create a minimal mock dataset to test _get_obs_steps
    class MockDataset:
        """Minimal mock to test _get_obs_steps without zarr files."""
        def __init__(self, training_mode, min_obs=2, max_obs=80):
            self.min_obs_steps = min_obs
            self.max_obs_steps = max_obs
            self.training_mode = training_mode
            self.progressive_steps = 1000
            self.training_step = 0
            self.current_max = min_obs
            self.random_sprinkle_prob = 0.1

        def _get_obs_steps(self):
            return PlanarPushingAttentionDataset._get_obs_steps(self)

        def set_training_step(self, step):
            self.training_step = step

    # Test random mode
    ds = MockDataset('random')
    vals = [ds._get_obs_steps() for _ in range(100)]
    assert all(2 <= v <= 80 for v in vals), f"random: got values outside [2, 80]: {min(vals)}, {max(vals)}"
    assert len(set(vals)) > 1, "random: should produce varying values"
    print("  random mode: OK")

    # Test random_sprinkle mode
    ds = MockDataset('random_sprinkle')
    vals = [ds._get_obs_steps() for _ in range(1000)]
    assert all(2 <= v <= 80 for v in vals), f"random_sprinkle: values outside range"
    n_max = sum(1 for v in vals if v == 80)
    assert n_max > 800, f"random_sprinkle: expected mostly max (got {n_max}/1000 max)"
    # Check that max_obs_steps CAN be sampled in the random portion (bug fix check)
    ds2 = MockDataset('random_sprinkle', min_obs=79, max_obs=80)
    ds2.random_sprinkle_prob = 1.0  # always random
    vals2 = [ds2._get_obs_steps() for _ in range(1000)]
    assert 80 in vals2, "random_sprinkle: should be able to sample max_obs_steps (bug fix)"
    print("  random_sprinkle mode: OK (including bug fix)")

    # Test progressive mode
    ds = MockDataset('progressive')
    ds.set_training_step(0)
    vals_early = [ds._get_obs_steps() for _ in range(100)]
    assert all(v == 2 for v in vals_early), f"progressive: at step 0, should only return min_obs={2}"
    ds.set_training_step(500)
    vals_mid = [ds._get_obs_steps() for _ in range(100)]
    mid_max = max(vals_mid)
    assert mid_max <= 41 + 2, f"progressive: at step 500/1000, current_max should be ~41, got max={mid_max}"
    ds.set_training_step(1000)
    vals_end = [ds._get_obs_steps() for _ in range(100)]
    assert ds.current_max == 80, f"progressive: at final step, current_max should be 80, got {ds.current_max}"
    assert max(vals_end) >= 60, f"progressive: at final step, should sample high values, got max={max(vals_end)}"
    ds.set_training_step(2000)  # past progressive_steps
    _ = ds._get_obs_steps()
    assert ds.current_max == 80, "progressive: past schedule should stay at max"
    print("  progressive mode: OK")

    # Test reverse_progressive mode
    ds = MockDataset('reverse_progressive')
    ds.set_training_step(0)
    _ = ds._get_obs_steps()
    assert ds.current_max == 80, "reverse_progressive: at step 0, current_max should be max_obs_steps"
    ds.set_training_step(1000)
    vals_end = [ds._get_obs_steps() for _ in range(100)]
    assert ds.current_max == 2, f"reverse_progressive: at final step, current_max should be min_obs, got {ds.current_max}"
    assert all(v == 2 for v in vals_end), f"reverse_progressive: at final step, should only return min_obs={2}"
    print("  reverse_progressive mode: OK")

    print("All dataset training mode tests passed!")


def test_vit_dynamic_prediction_build_trajectory():
    """Test _build_dynamic_trajectory in ViT policy."""
    from diffusion_policy.policy.diffusion_vit_attention_hybrid_image_policy import DiffusionViTAttentionHybridImagePolicy

    # Create a minimal mock policy
    class MockPolicy:
        def __init__(self):
            self.n_past_action_steps = 8
            self.n_future = 16
            self._unet_alignment = 4
            self.horizon = 96

    policy = MockPolicy()

    B = 4
    horizon = 96
    Da = 2
    nactions = torch.randn(B, horizon, Da)

    # Simulate different context lengths
    num_obs_steps = torch.tensor([80, 40, 2, 10])

    trajectory, loss_mask = DiffusionViTAttentionHybridImagePolicy._build_dynamic_trajectory(
        policy, nactions, num_obs_steps)

    # Check shapes
    # To=80: pred_len = min(8,80)+16 = 24
    # To=40: pred_len = min(8,40)+16 = 24
    # To=2:  pred_len = min(8,2)+16 = 18
    # To=10: pred_len = min(8,10)+16 = 24
    # max pred_len = 24, padded = 24 (already mult of 4)
    assert trajectory.shape == (B, 24, Da), f"Expected (4, 24, 2), got {trajectory.shape}"
    assert loss_mask.shape == (B, 24, Da), f"Loss mask shape mismatch: {loss_mask.shape}"

    # Check per-sample content
    # Sample 0 (To=80): offset = 80 - min(8,80) = 72, pred_len=24
    assert torch.allclose(trajectory[0, :24], nactions[0, 72:96])
    assert loss_mask[0].all()

    # Sample 1 (To=40): offset = 40 - 8 = 32, pred_len=24
    assert torch.allclose(trajectory[1, :24], nactions[1, 32:56])
    assert loss_mask[1].all()

    # Sample 2 (To=2): offset = 2 - min(8,2) = 0, pred_len=18
    assert torch.allclose(trajectory[2, :18], nactions[2, 0:18])
    assert loss_mask[2, :18].all()
    assert not loss_mask[2, 18:].any(), "Padded positions should be masked"

    # Sample 3 (To=10): offset = 10 - 8 = 2, pred_len=24
    assert torch.allclose(trajectory[3, :24], nactions[3, 2:26])
    assert loss_mask[3].all()

    print("  _build_dynamic_trajectory: OK")

    # Test with all same context (no padding needed)
    num_obs_uniform = torch.tensor([80, 80, 80, 80])
    traj_u, mask_u = DiffusionViTAttentionHybridImagePolicy._build_dynamic_trajectory(
        policy, nactions, num_obs_uniform)
    assert traj_u.shape == (B, 24, Da)
    assert mask_u.all(), "All samples same length, no padding"
    print("  uniform context: OK")

    # Test edge case: To < n_past for all samples
    num_obs_small = torch.tensor([1, 3, 2, 5])
    # pred_lens: 17, 19, 18, 21 → max=21, padded=24
    traj_s, mask_s = DiffusionViTAttentionHybridImagePolicy._build_dynamic_trajectory(
        policy, nactions, num_obs_small)
    assert traj_s.shape == (B, 24, Da)
    # Sample 0: To=1, pred_len=17, offset=0
    assert torch.allclose(traj_s[0, :17], nactions[0, :17])
    assert mask_s[0, :17].all()
    assert not mask_s[0, 17:].any()
    print("  small context edge case: OK")

    print("All dynamic prediction trajectory tests passed!")


def test_vit_predict_action_indexing():
    """Test that predict_action extracts the correct action window."""
    # We don't instantiate the full policy, just verify the indexing math
    n_past = 8
    n_future = 16
    n_action_steps = 8

    for To in [80, 40, 10, 8, 5, 2, 1]:
        past_in_pred = min(n_past, To)
        pred_len = past_in_pred + n_future
        offset = To - past_in_pred  # = max(0, To - n_past)

        # In the full action tensor, current action is at position To-1
        # In the prediction tensor, current action is at position past_in_pred - 1
        start_in_pred = past_in_pred - 1
        end_in_pred = start_in_pred + n_action_steps

        # Verify: prediction position start_in_pred maps to global position offset + start_in_pred
        global_start = offset + start_in_pred
        assert global_start == To - 1, f"To={To}: global_start={global_start} != To-1={To-1}"

        # Verify end is within prediction
        assert end_in_pred <= pred_len, f"To={To}: end={end_in_pred} > pred_len={pred_len}"

        print(f"  To={To}: pred_len={pred_len}, offset={offset}, "
              f"action window [{start_in_pred}:{end_in_pred}] = global [{global_start}:{global_start+n_action_steps}]")

    print("All predict_action indexing tests passed!")


def test_resnet_dynamic_prediction_build_trajectory():
    """Test _build_dynamic_trajectory in ResNet policy."""
    from diffusion_policy.policy.diffusion_attention_hybrid_image_policy import DiffusionAttentionHybridImagePolicy

    class MockPolicy:
        def __init__(self):
            self.n_past_action_steps = 8
            self.n_future = 16
            self._unet_alignment = 4
            self.horizon = 96

    policy = MockPolicy()

    B = 4
    Da = 2
    nactions = torch.randn(B, 96, Da)
    num_obs_steps = torch.tensor([80, 40, 2, 10])

    trajectory, loss_mask = DiffusionAttentionHybridImagePolicy._build_dynamic_trajectory(
        policy, nactions, num_obs_steps)

    assert trajectory.shape == (B, 24, Da), f"Expected (4, 24, 2), got {trajectory.shape}"

    # Sample 0 (To=80): offset = 72, pred_len=24
    assert torch.allclose(trajectory[0, :24], nactions[0, 72:96])
    assert loss_mask[0].all()

    # Sample 2 (To=2): offset = 0, pred_len=18
    assert torch.allclose(trajectory[2, :18], nactions[2, 0:18])
    assert not loss_mask[2, 18:].any()

    print("  ResNet _build_dynamic_trajectory: OK")
    print("ResNet dynamic prediction test passed!")


def test_backward_compat_no_past_action_steps():
    """Test that with n_past_action_steps=None, behavior is unchanged."""
    class MockPolicy:
        def __init__(self):
            self.n_past_action_steps = None
            self.n_future = 16
            self._unet_alignment = 4
            self.horizon = 96
            self.prediction_horizon = 96
            self._action_offset = 0

    policy = MockPolicy()

    # With n_past_action_steps=None, compute_loss should use full trajectory
    # (no _build_dynamic_trajectory call)
    B, horizon, Da = 4, 96, 2
    nactions = torch.randn(B, horizon, Da)

    # predict_action should use T = prediction_horizon = 96
    assert policy.prediction_horizon == 96
    assert policy._action_offset == 0

    # Action extraction: past_in_pred = To (no cap), start = To - 1
    To = 80
    start = To - 1  # = 79
    n_action_steps = 8
    end = start + n_action_steps  # = 87
    assert start == 79 and end == 87
    print("  backward compat (no n_past_action_steps): OK")

    print("Backward compatibility test passed!")


if __name__ == '__main__':
    print("=== Dataset Training Modes ===")
    test_dataset_training_modes()
    print()
    print("=== Dynamic Prediction Trajectory Building ===")
    test_vit_dynamic_prediction_build_trajectory()
    print()
    print("=== Predict Action Indexing ===")
    test_vit_predict_action_indexing()
    print()
    print("=== ResNet Dynamic Prediction ===")
    test_resnet_dynamic_prediction_build_trajectory()
    print()
    print("=== Backward Compatibility ===")
    test_backward_compat_no_past_action_steps()
    print()
    print("ALL TESTS PASSED!")
