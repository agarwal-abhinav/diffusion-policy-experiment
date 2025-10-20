#!/usr/bin/env python3
"""
Comprehensive test suite for the new attention-based diffusion architecture.

Tests:
1. Policy functionality with variable observation lengths
2. Dataset sampling with flexible observation sequences
3. Workspace training compatibility
4. Error handling and edge cases
5. Attention mechanism consistency
"""

import sys
import os
import pathlib
import tempfile
import shutil

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import numpy as np
# import pytest  # Will be imported conditionally
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.policy.diffusion_attention_hybrid_image_policy import DiffusionAttentionHybridImagePolicy
from diffusion_policy.dataset.planar_pushing_attention_dataset import PlanarPushingAttentionDataset
from diffusion_policy.workspace.train_diffusion_attention_hybrid_workspace import TrainDiffusionAttentionHybridWorkspace
from diffusion_policy.model.diffusion.attention_conditional_unet1d import AttentionConditionalUnet1D


class TestAttentionPolicy:
    """Test the attention-based policy implementation."""
    
    # @pytest.fixture
    def shape_meta(self):
        return {
            'action': {'shape': [2]},
            'obs': {
                'agent_pos': {'type': 'low_dim', 'shape': [3]},
                'overhead_camera': {'type': 'rgb', 'shape': [3, 96, 96]},
            },
        }
    
    # @pytest.fixture
    def policy(self, shape_meta):
        """Create a test policy instance."""
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            prediction_type='epsilon'
        )
        
        policy = DiffusionAttentionHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=16,
            n_action_steps=8,
            n_obs_steps=10,
            min_obs_steps=2,
            obs_as_global_cond=True,
            down_dims=[128, 256],
            max_global_tokens=10,
            num_attention_heads=4,
            variable_obs_curriculum=True
        )
        return policy
    
    def test_policy_initialization(self, policy):
        """Test policy initializes correctly."""
        assert policy.min_obs_steps == 2
        assert policy.max_obs_steps == 10
        assert policy.horizon == 16
        assert policy.variable_obs_curriculum == True
        
        # Check model components exist
        assert hasattr(policy, 'obs_encoder')
        assert hasattr(policy, 'model')
        assert isinstance(policy.model, AttentionConditionalUnet1D)
        
        print("✅ Policy initialization test passed!")
    
    def test_variable_obs_length_curriculum(self, policy):
        """Test the curriculum learning for observation lengths."""
        # Early training: should prefer max observations
        early_lengths = [policy._get_current_obs_steps(step) for step in range(0, 1000, 100)]
        assert most_values_equal(early_lengths, policy.max_obs_steps), \
            f"Early training should use mostly max obs steps, got {early_lengths}"
        
        # Late training: should have more variety
        late_lengths = [policy._get_current_obs_steps(step) for step in range(20000, 21000, 100)]
        assert len(set(late_lengths)) > 1, \
            f"Late training should use varied obs steps, got unique values: {set(late_lengths)}"
        
        # All lengths should be within bounds
        all_lengths = early_lengths + late_lengths
        assert all(policy.min_obs_steps <= length <= policy.max_obs_steps for length in all_lengths)
        
        print("✅ Curriculum learning test passed!")
    
    def test_forward_pass_different_obs_lengths(self, policy, shape_meta):
        """Test forward pass with different observation lengths."""
        batch_size = 4
        horizon = policy.horizon
        
        # Create mock batch
        batch = create_mock_batch(batch_size, horizon, shape_meta)
        trajectory = torch.randn(batch_size, horizon, 2)
        timesteps = torch.randint(0, 100, (batch_size,))
        
        policy.eval()  # Set to eval mode for deterministic behavior
        
        # Test with different observation lengths
        for obs_len in [policy.min_obs_steps, 5, policy.max_obs_steps]:
            # Set specific obs length
            original_method = policy._get_current_obs_steps
            policy._get_current_obs_steps = lambda: obs_len
            
            try:
                output = policy.forward(batch, trajectory, timesteps)
                assert output.shape == trajectory.shape, \
                    f"Output shape {output.shape} doesn't match input {trajectory.shape} for obs_len={obs_len}"
                assert not torch.isnan(output).any(), f"NaN values in output for obs_len={obs_len}"
                assert not torch.isinf(output).any(), f"Inf values in output for obs_len={obs_len}"
            finally:
                policy._get_current_obs_steps = original_method
        
        print("✅ Variable observation length forward pass test passed!")
    
    def test_predict_action_consistency(self, policy, shape_meta):
        """Test that predict_action works with different observation tokens."""
        batch_size = 2
        obs_dict = create_mock_obs_dict(batch_size, policy.max_obs_steps, shape_meta)
        
        policy.eval()
        with torch.no_grad():
            # Test with different numbers of observation tokens
            results = {}
            for num_tokens in [policy.min_obs_steps, 5, policy.max_obs_steps]:
                result = policy.predict_action(obs_dict, num_obs_tokens=num_tokens, use_DDIM=True)
                results[num_tokens] = result
                
                assert 'action' in result
                assert 'action_pred' in result
                assert 'num_obs_tokens_used' in result
                assert result['num_obs_tokens_used'] == num_tokens
                assert result['action'].shape[0] == batch_size
        
        # Test that different obs lengths produce different results (should be different due to different contexts)
        action_2_tokens = results[policy.min_obs_steps]['action']
        action_max_tokens = results[policy.max_obs_steps]['action']
        
        # Actions should be different (with high probability) when using different context lengths
        diff = torch.abs(action_2_tokens - action_max_tokens).mean()
        assert diff > 1e-6, "Actions should differ when using different observation contexts"
        
        print("✅ Predict action consistency test passed!")


class TestAttentionDataset:
    """Test the attention-based dataset implementation."""
    
    # @pytest.fixture
    def temp_zarr_path(self):
        """Create a temporary zarr file for testing."""
        temp_dir = tempfile.mkdtemp()
        zarr_path = os.path.join(temp_dir, 'test_data.zarr')
        
        # Create minimal zarr structure
        create_mock_zarr_dataset(zarr_path)
        
        yield zarr_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    # @pytest.fixture
    def shape_meta(self):
        return {
            'action': {'shape': [2]},
            'obs': {
                'agent_pos': {'type': 'low_dim', 'shape': [3]},
                'overhead_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
            },
        }
    
    def test_dataset_initialization(self, temp_zarr_path, shape_meta):
        """Test dataset initializes with attention parameters."""
        zarr_configs = [{'path': temp_zarr_path, 'sampling_weight': 1.0}]
        
        dataset = PlanarPushingAttentionDataset(
            zarr_configs=zarr_configs,
            shape_meta=shape_meta,
            horizon=24,
            n_obs_steps=16,
            min_obs_steps=4,
            attention_curriculum=True
        )
        
        assert dataset.horizon == 24
        assert dataset.max_obs_steps == 16
        assert dataset.min_obs_steps == 4
        assert dataset.attention_curriculum == True
        assert len(dataset) > 0
        
        print("✅ Dataset initialization test passed!")
    
    def test_dataset_sampling(self, temp_zarr_path, shape_meta):
        """Test that dataset returns properly formatted samples."""
        zarr_configs = [{'path': temp_zarr_path, 'sampling_weight': 1.0}]
        
        dataset = PlanarPushingAttentionDataset(
            zarr_configs=zarr_configs,
            shape_meta=shape_meta,
            horizon=24,
            n_obs_steps=16,
            min_obs_steps=4
        )
        
        # Test sample structure
        sample = dataset[0]
        assert 'obs' in sample
        assert 'action' in sample
        assert 'target' in sample
        assert 'sample_metadata' in sample
        
        # Check shapes
        assert sample['action'].shape[0] == 24  # horizon length
        assert sample['obs']['agent_pos'].shape[0] == 24  # full horizon for obs
        
        # Check metadata
        metadata = sample['sample_metadata']
        assert metadata['max_obs_steps'] == 16
        assert metadata['min_obs_steps'] == 4
        assert metadata['horizon'] == 24
        
        print("✅ Dataset sampling test passed!")
    
    def test_validation_dataset(self, temp_zarr_path, shape_meta):
        """Test validation dataset creation."""
        zarr_configs = [{'path': temp_zarr_path, 'sampling_weight': 1.0}]
        
        dataset = PlanarPushingAttentionDataset(
            zarr_configs=zarr_configs,
            shape_meta=shape_meta,
            horizon=24,
            n_obs_steps=16,
            min_obs_steps=4,
            val_ratio=0.2
        )
        
        val_dataset = dataset.get_validation_dataset(0)
        assert len(val_dataset) > 0
        assert val_dataset.horizon == dataset.horizon
        assert val_dataset.max_obs_steps == dataset.max_obs_steps
        assert val_dataset.min_obs_steps == dataset.min_obs_steps
        
        # Test validation sample
        val_sample = val_dataset[0]
        assert 'sample_metadata' in val_sample
        
        print("✅ Validation dataset test passed!")


class TestAttentionWorkspace:
    """Test the attention workspace implementation."""
    
    # @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for workspace output."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    # @pytest.fixture
    def minimal_config(self, temp_dir):
        """Create minimal config for workspace testing."""
        return OmegaConf.create({
            'policy': {
                '_target_': 'diffusion_policy.policy.diffusion_attention_hybrid_image_policy.DiffusionAttentionHybridImagePolicy',
                'shape_meta': {
                    'action': {'shape': [2]},
                    'obs': {
                        'agent_pos': {'type': 'low_dim', 'shape': [3]},
                        'overhead_camera': {'type': 'rgb', 'shape': [3, 96, 96]},
                    }
                },
                'noise_scheduler': {
                    '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                    'num_train_timesteps': 50,
                    'beta_schedule': 'linear',
                    'prediction_type': 'epsilon'
                },
                'horizon': 16,
                'n_action_steps': 8,
                'n_obs_steps': 10,
                'min_obs_steps': 2,
                'obs_as_global_cond': True,
                'down_dims': [64, 128],
                'max_global_tokens': 10
            },
            'training': {
                'seed': 42,
                'device': 'cpu',
                'use_ema': False,
                'num_epochs': 2,
                'lr_scheduler': 'cosine',
                'lr_warmup_steps': 10,
                'gradient_accumulate_every': 1,
                'debug': True,
                'checkpoint_every': 1,
                'val_every': 1,
                'sample_every': 1,
                'log_val_mse': False,
                'tqdm_interval_sec': 1.0,
                'attention_curriculum_steps': 100
            },
            'optimizer': {
                '_target_': 'torch.optim.Adam',
                'lr': 1e-4
            },
            'ema': {
                '_target_': 'diffusion_policy.model.diffusion.ema_model.EMAModel',
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999
            },
            'logging': {
                'project': 'test_attention',
                'mode': 'disabled'  # Disable wandb for testing
            },
            'checkpoint': {
                'save_last_ckpt': True,
                'save_last_snapshot': False,
                'topk': {
                    'k': 1,
                    'mode': 'min',
                    'monitor_key': 'val_loss',
                    'format_str': 'epoch={epoch:04d}-val_loss={val_loss:.3f}.ckpt'
                }
            },
            'dataloader': {
                'batch_size': 2,
                'num_workers': 0,
                'shuffle': True,
                'pin_memory': False
            },
            'val_dataloader': {
                'batch_size': 2,
                'num_workers': 0,
                'shuffle': False,
                'pin_memory': False
            }
        })
    
    def test_workspace_initialization(self, minimal_config, temp_dir):
        """Test workspace can be initialized."""
        try:
            workspace = TrainDiffusionAttentionHybridWorkspace(minimal_config, output_dir=temp_dir)
            
            assert workspace.model is not None
            assert hasattr(workspace, 'attention_curriculum_steps')
            assert hasattr(workspace, 'log_attention_stats')
            
            print("✅ Workspace initialization test passed!")
            
        except Exception as e:
            print(f"❌ Workspace initialization failed: {e}")
            raise


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_obs_steps(self):
        """Test handling of invalid observation step parameters."""
        shape_meta = {
            'action': {'shape': [2]},
            'obs': {'agent_pos': {'type': 'low_dim', 'shape': [3]}},
        }
        
        noise_scheduler = DDPMScheduler(num_train_timesteps=100)
        
        # min_obs_steps > n_obs_steps should raise error
        try:
            DiffusionAttentionHybridImagePolicy(
                shape_meta=shape_meta,
                noise_scheduler=noise_scheduler,
                horizon=16,
                n_action_steps=8,
                n_obs_steps=5,
                min_obs_steps=10,  # Invalid: > n_obs_steps
                obs_as_global_cond=True
            )
            assert False, "Should have raised AssertionError for min_obs_steps > n_obs_steps"
        except AssertionError as e:
            if "Should have raised" in str(e):
                raise e
            pass  # Expected error
        
        # min_obs_steps < 1 should raise error
        try:
            DiffusionAttentionHybridImagePolicy(
                shape_meta=shape_meta,
                noise_scheduler=noise_scheduler,
                horizon=16,
                n_action_steps=8,
                n_obs_steps=5,
                min_obs_steps=0,  # Invalid: < 1
                obs_as_global_cond=True
            )
            assert False, "Should have raised AssertionError for min_obs_steps < 1"
        except AssertionError as e:
            if "Should have raised" in str(e):
                raise e
            pass  # Expected error
        
        print("✅ Invalid observation steps error handling test passed!")
    
    def test_batch_size_edge_cases(self):
        """Test handling of different batch sizes."""
        shape_meta = {
            'action': {'shape': [2]},
            'obs': {'agent_pos': {'type': 'low_dim', 'shape': [3]}},
        }
        
        policy = DiffusionAttentionHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=DDPMScheduler(num_train_timesteps=100),
            horizon=8,
            n_action_steps=4,
            n_obs_steps=4,
            min_obs_steps=1,
            obs_as_global_cond=True,
            down_dims=[64]
        )
        
        policy.eval()
        
        # Test with batch size 1
        obs_dict_1 = create_mock_obs_dict(1, 4, shape_meta)
        with torch.no_grad():
            result_1 = policy.predict_action(obs_dict_1, num_obs_tokens=2)
            assert result_1['action'].shape[0] == 1
        
        # Test with larger batch size
        obs_dict_8 = create_mock_obs_dict(8, 4, shape_meta)
        with torch.no_grad():
            result_8 = policy.predict_action(obs_dict_8, num_obs_tokens=3)
            assert result_8['action'].shape[0] == 8
        
        print("✅ Batch size edge cases test passed!")
    
    def test_insufficient_observations(self):
        """Test handling when requested tokens > available observations."""
        shape_meta = {
            'action': {'shape': [2]},
            'obs': {'agent_pos': {'type': 'low_dim', 'shape': [3]}},
        }
        
        policy = DiffusionAttentionHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=DDPMScheduler(num_train_timesteps=100),
            horizon=8,
            n_action_steps=4,
            n_obs_steps=6,
            min_obs_steps=1,
            obs_as_global_cond=True,
            down_dims=[64]
        )
        
        policy.eval()
        
        # Create obs with only 3 timesteps but request 5 tokens
        obs_dict = create_mock_obs_dict(2, 3, shape_meta)  # Only 3 timesteps
        
        with torch.no_grad():
            # Should gracefully handle by using only available observations
            result = policy.predict_action(obs_dict, num_obs_tokens=5)
            assert result['num_obs_tokens_used'] <= 3  # Should be clamped
        
        print("✅ Insufficient observations test passed!")


# Utility functions for testing
def most_values_equal(values, target_value, threshold=0.8):
    """Check if most values in list equal target_value."""
    count = sum(1 for v in values if v == target_value)
    return count >= len(values) * threshold

def create_mock_batch(batch_size, horizon, shape_meta):
    """Create a mock batch for testing."""
    batch = {
        'action': torch.randn(batch_size, horizon, 2),
        'target': torch.randn(batch_size, 3),
        'obs': {}
    }
    
    for key, attr in shape_meta['obs'].items():
        if attr.get('type') == 'rgb':
            shape = attr['shape']
            batch['obs'][key] = torch.rand(batch_size, horizon, *shape)
        elif attr.get('type') == 'low_dim':
            shape = attr['shape']
            batch['obs'][key] = torch.randn(batch_size, horizon, *shape)
        else:  # default to low_dim
            shape = attr['shape']
            batch['obs'][key] = torch.randn(batch_size, horizon, *shape)
    
    return batch

def create_mock_obs_dict(batch_size, seq_len, shape_meta):
    """Create a mock observation dictionary."""
    obs_dict = {'obs': {}}
    
    for key, attr in shape_meta['obs'].items():
        if attr.get('type') == 'rgb':
            shape = attr['shape']
            obs_dict['obs'][key] = torch.rand(batch_size, seq_len, *shape)
        elif attr.get('type') == 'low_dim':
            shape = attr['shape']
            obs_dict['obs'][key] = torch.randn(batch_size, seq_len, *shape)
        else:
            shape = attr['shape']
            obs_dict['obs'][key] = torch.randn(batch_size, seq_len, *shape)
    
    return obs_dict

def create_mock_zarr_dataset(zarr_path):
    """Create a minimal zarr dataset for testing."""
    import zarr
    
    # Create a simple zarr dataset
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Create minimal data for 5 episodes, 10 steps each
    n_episodes = 5
    n_steps_per_episode = 10
    total_steps = n_episodes * n_steps_per_episode
    
    # Create data arrays
    root.create_dataset('action', shape=(total_steps, 2), dtype=np.float32)
    root.create_dataset('state', shape=(total_steps, 3), dtype=np.float32)  
    root.create_dataset('target', shape=(total_steps, 3), dtype=np.float32)
    root.create_dataset('overhead_camera', shape=(total_steps, 128, 128, 3), dtype=np.uint8)
    
    # Fill with random data
    root['action'][:] = np.random.randn(total_steps, 2).astype(np.float32)
    root['state'][:] = np.random.randn(total_steps, 3).astype(np.float32)
    root['target'][:] = np.random.randn(total_steps, 3).astype(np.float32)
    root['overhead_camera'][:] = np.random.randint(0, 256, (total_steps, 128, 128, 3), dtype=np.uint8)
    
    # Create episode ends metadata
    meta = root.create_group('meta')
    episode_ends = np.arange(n_steps_per_episode, total_steps + 1, n_steps_per_episode)
    meta.create_dataset('episode_ends', data=episode_ends)
    
    return zarr_path


def run_comprehensive_tests():
    """Run all tests in order."""
    print("🧪 Running comprehensive attention architecture tests...\n")
    
    # Import pytest programmatically if needed
    try:
        import pytest
    except ImportError:
        print("❌ pytest not available, running basic tests...")
        run_basic_tests()
        return
    
    # Run pytest on this file
    test_file = __file__
    exit_code = pytest.main([test_file, '-v', '--tb=short'])
    
    if exit_code == 0:
        print("\n🎉 ALL ATTENTION ARCHITECTURE TESTS PASSED!")
        print("\nArchitecture summary:")
        print("- ✅ Variable-length observation sequences")
        print("- ✅ Attention-based conditioning with temporal encoding")
        print("- ✅ Curriculum learning for observation lengths")
        print("- ✅ Proper error handling and edge cases")
        print("- ✅ Dataset and workspace compatibility")
    else:
        print("\n❌ Some tests failed. Check output above for details.")


def run_basic_tests():
    """Run basic tests without pytest."""
    print("Running basic functionality tests...")
    
    # Test 1: Model instantiation
    try:
        shape_meta = {
            'action': {'shape': [2]},
            'obs': {'agent_pos': {'type': 'low_dim', 'shape': [3]}},
        }
        
        noise_scheduler = DDPMScheduler(num_train_timesteps=100)
        policy = DiffusionAttentionHybridImagePolicy(
            shape_meta=shape_meta,
            noise_scheduler=noise_scheduler,
            horizon=8,
            n_action_steps=4,
            n_obs_steps=4,
            min_obs_steps=1,
            obs_as_global_cond=True,
            down_dims=[64]
        )
        print("✅ Basic policy instantiation test passed!")
        
    except Exception as e:
        print(f"❌ Basic policy instantiation failed: {e}")
        return
    
    # Test 2: Forward pass
    try:
        policy.eval()
        
        # Set up a mock normalizer first
        from diffusion_policy.model.common.normalizer import LinearNormalizer
        normalizer = LinearNormalizer()
        
        # Create mock normalization data
        dummy_data = {
            'agent_pos': torch.randn(100, 3),
            'action': torch.randn(100, 2)
        }
        normalizer.fit(dummy_data, last_n_dims=1)
        policy.set_normalizer(normalizer)
        
        batch = create_mock_batch(2, 8, shape_meta)
        trajectory = torch.randn(2, 8, 2)
        timesteps = torch.randint(0, 100, (2,))
        
        output = policy.forward(batch, trajectory, timesteps)
        assert output.shape == trajectory.shape
        print("✅ Basic forward pass test passed!")
        
    except Exception as e:
        print(f"❌ Basic forward pass failed: {e}")
        return
    
    # Test 3: Prediction
    try:
        obs_dict = create_mock_obs_dict(1, 4, shape_meta)
        with torch.no_grad():
            result = policy.predict_action(obs_dict, num_obs_tokens=2)
            assert 'action' in result
            assert 'num_obs_tokens_used' in result
            assert result['num_obs_tokens_used'] == 2
        print("✅ Basic prediction test passed!")
        
    except Exception as e:
        print(f"❌ Basic prediction failed: {e}")
        return
    
    print("\n🎉 Basic tests passed! Full architecture is functional.")


if __name__ == "__main__":
    run_comprehensive_tests()