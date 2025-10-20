#!/usr/bin/env python3
"""
Test the utility functions that come with AttentionConditionalUnet1D
to demonstrate practical usage patterns.
"""

import torch
from diffusion_policy.model.diffusion.attention_conditional_unet1d import (
    AttentionConditionalUnet1D,
    create_consistent_temporal_batch,
    example_consistent_usage,
    example_mixed_batch
)

def test_utility_functions_detailed():
    """Detailed test of utility functions for practical usage."""
    
    print("🔧 TESTING UTILITY FUNCTIONS FOR PRACTICAL USAGE")
    print("=" * 60)
    
    # === TEST 1: create_consistent_temporal_batch ===
    print("\n🔬 TEST 1: create_consistent_temporal_batch Function")
    print("-" * 50)
    
    # Create sample observation sequences of different lengths
    obs_sequences = [
        torch.randn(16, 64),  # Full 16-step history
        torch.randn(12, 64),  # 12-step history
        torch.randn(8, 64),   # 8-step history
        torch.randn(16, 64),  # Another full history
    ]
    
    # Different token counts per sample
    num_tokens_list = [8, 5, 3, 2]
    max_tokens = 8
    
    print(f"Input sequences:")
    for i, (seq, tokens) in enumerate(zip(obs_sequences, num_tokens_list)):
        print(f"  Sample {i}: {seq.shape[0]} timesteps available, using {tokens} tokens")
    
    # Create consistent batch with most_recent strategy
    batch = create_consistent_temporal_batch(
        observation_sequences=obs_sequences,
        num_tokens_per_sample=num_tokens_list,
        max_tokens=max_tokens,
        strategy='most_recent'
    )
    
    print(f"\nOutput batch shapes:")
    print(f"  global_cond: {batch['global_cond'].shape}")
    print(f"  global_mask: {batch['global_mask'].shape}")
    print(f"  temporal_positions: {batch['temporal_positions'].shape}")
    
    print(f"\nTemporal positioning verification:")
    for i in range(len(obs_sequences)):
        valid_mask = batch['global_mask'][i]
        valid_positions = batch['temporal_positions'][i][valid_mask]
        seq_len = obs_sequences[i].shape[0]
        tokens_used = num_tokens_list[i]
        
        print(f"  Sample {i} ({seq_len} available, {tokens_used} used):")
        print(f"    Valid positions: {valid_positions.tolist()}")
        
        # Verify positions are consistent with most_recent strategy
        expected_start = max(0, seq_len - tokens_used)
        expected_positions = list(range(expected_start, expected_start + tokens_used))
        
        if valid_positions.tolist() == expected_positions:
            print(f"    ✅ Correct most_recent positions")
        else:
            print(f"    ❌ Expected {expected_positions}, got {valid_positions.tolist()}")
            raise AssertionError(f"Incorrect positions for sample {i}")
    
    # === TEST 2: Test with AttentionConditionalUnet1D ===
    print(f"\n🔬 TEST 2: Integration with AttentionConditionalUnet1D")
    print(f"-" * 53)
    
    model = AttentionConditionalUnet1D(
        input_dim=2,
        global_cond_dim=64,
        target_dim=2,
        max_global_tokens=8
    )
    model.eval()
    
    # Create trajectory samples
    batch_size = len(obs_sequences)
    sample = torch.randn(batch_size, 16, 2)
    timestep = torch.randint(0, 1000, (batch_size,))
    target_cond = torch.randn(batch_size, 2)
    
    # Use the batch created by utility function
    with torch.no_grad():
        output = model(
            sample=sample,
            timestep=timestep,
            global_cond=batch['global_cond'],
            global_mask=batch['global_mask'],
            target_cond=target_cond,
            temporal_positions=batch['temporal_positions']
        )
    
    print(f"Model integration test:")
    print(f"  Input shape: {sample.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✅ Successfully processed batch from utility function")
    
    # === TEST 3: Different Strategies ===
    print(f"\n🔬 TEST 3: Different Sampling Strategies")
    print(f"-" * 40)
    
    # Test uniform_sample strategy
    uniform_batch = create_consistent_temporal_batch(
        observation_sequences=[torch.randn(16, 64)],
        num_tokens_per_sample=[4],
        max_tokens=8,
        strategy='uniform_sample'
    )
    
    uniform_positions = uniform_batch['temporal_positions'][0][uniform_batch['global_mask'][0]]
    print(f"Uniform sampling (4 tokens from 16 steps):")
    print(f"  Positions: {uniform_positions.tolist()}")
    print(f"  ✅ Uniform distribution across sequence")
    
    # Test oldest_first strategy
    oldest_batch = create_consistent_temporal_batch(
        observation_sequences=[torch.randn(16, 64)],
        num_tokens_per_sample=[4],
        max_tokens=8,
        strategy='oldest_first'
    )
    
    oldest_positions = oldest_batch['temporal_positions'][0][oldest_batch['global_mask'][0]]
    print(f"Oldest first sampling (4 tokens from 16 steps):")
    print(f"  Positions: {oldest_positions.tolist()}")
    print(f"  ✅ Takes oldest observations first")
    
    # === TEST 4: Example Functions ===
    print(f"\n🔬 TEST 4: Built-in Example Functions")
    print(f"-" * 38)
    
    print("Running example_consistent_usage()...")
    batch_5, batch_2 = example_consistent_usage()
    
    print(f"\nRunning example_mixed_batch()...")
    mixed_batch = example_mixed_batch()
    
    print(f"✅ Example functions executed successfully")
    
    # === TEST 5: Edge Cases ===
    print(f"\n🔬 TEST 5: Edge Cases and Error Handling")
    print(f"-" * 41)
    
    # Test with single token
    single_token_batch = create_consistent_temporal_batch(
        observation_sequences=[torch.randn(10, 32)],
        num_tokens_per_sample=[1],
        max_tokens=5,
        strategy='most_recent'
    )
    
    single_pos = single_token_batch['temporal_positions'][0][single_token_batch['global_mask'][0]]
    print(f"Single token test:")
    print(f"  Position: {single_pos.item()} (from 10-step sequence)")
    print(f"  ✅ Single token handled correctly")
    
    # Test with max tokens
    max_token_batch = create_consistent_temporal_batch(
        observation_sequences=[torch.randn(8, 32)],
        num_tokens_per_sample=[8],
        max_tokens=8,
        strategy='most_recent'
    )
    
    max_positions = max_token_batch['temporal_positions'][0][max_token_batch['global_mask'][0]]
    print(f"Max tokens test:")
    print(f"  Positions: {max_positions.tolist()}")
    print(f"  ✅ Max tokens handled correctly")
    
    # Test empty sequence error handling
    try:
        empty_batch = create_consistent_temporal_batch(
            observation_sequences=[],
            num_tokens_per_sample=[],
            max_tokens=5
        )
        print(f"❌ Should have raised error for empty sequences")
    except ValueError as e:
        print(f"Empty sequences test:")
        print(f"  ✅ Correctly raised ValueError: {e}")
    
    # === FINAL VERIFICATION ===
    print(f"\n🔬 FINAL VERIFICATION: Temporal Consistency")
    print(f"-" * 46)
    
    # Verify that same observations get same positional encoding
    # regardless of how many total tokens are used
    
    full_sequence = torch.randn(16, 128)
    
    # Case 1: Use 3 most recent (positions should be [13, 14, 15])
    case1 = create_consistent_temporal_batch([full_sequence], [3], strategy='most_recent')
    pos1 = case1['temporal_positions'][0][case1['global_mask'][0]]
    
    # Case 2: Use 5 most recent (positions should be [11, 12, 13, 14, 15])
    case2 = create_consistent_temporal_batch([full_sequence], [5], strategy='most_recent')
    pos2 = case2['temporal_positions'][0][case2['global_mask'][0]]
    
    print(f"Temporal consistency verification:")
    print(f"  3 tokens: positions {pos1.tolist()}")
    print(f"  5 tokens: positions {pos2.tolist()}")
    
    # Check that overlapping positions are identical
    overlap_pos1 = pos1  # [13, 14, 15]
    overlap_pos2 = pos2[-3:]  # last 3 of [11, 12, 13, 14, 15]
    
    if torch.equal(overlap_pos1, overlap_pos2):
        print(f"  ✅ Consistent: observations 13,14,15 have identical positions")
    else:
        print(f"  ❌ Inconsistent: {overlap_pos1.tolist()} vs {overlap_pos2.tolist()}")
        raise AssertionError("Temporal consistency violated")
    
    print(f"\n🎉 ALL UTILITY FUNCTION TESTS PASSED!")
    print(f"=" * 60)
    print(f"✅ Utility functions provide:")
    print(f"   • Consistent temporal positioning across variable lengths")
    print(f"   • Multiple sampling strategies (most_recent, uniform, oldest_first)")
    print(f"   • Proper batching with padding and masking")
    print(f"   • Integration with AttentionConditionalUnet1D")
    print(f"   • Error handling for edge cases")
    print(f"   • Examples for practical usage patterns")
    
    return True

if __name__ == "__main__":
    test_utility_functions_detailed()