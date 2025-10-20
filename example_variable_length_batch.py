#!/usr/bin/env python3
"""
Practical code example showing how to create variable-length batches
for AttentionConditionalUnet1D exactly as shown in the test results.
"""

import torch
from diffusion_policy.model.diffusion.attention_conditional_unet1d import AttentionConditionalUnet1D

def create_variable_length_batch_example():
    """
    Create the exact variable-length batch as shown in test results:
    Sample 0: 8 tokens → [0,1,2,3,4,5,6,7] (8 steps ago to most recent)
    Sample 1: 5 tokens → [3,4,5,6,7] (5 steps ago to most recent)  
    Sample 2: 3 tokens → [5,6,7] (3 steps ago to most recent)
    Sample 3: 2 tokens → [6,7] (2 steps ago to most recent)
    """
    
    print("🔧 CREATING VARIABLE-LENGTH BATCH EXAMPLE")
    print("=" * 60)
    
    # Configuration matching your requirements
    batch_size = 4
    horizon = 16  # Trajectory length
    input_dim = 2  # Action dimension
    global_cond_dim = 256  # Observation feature dimension
    target_dim = 2  # Target/goal dimension
    max_obs_steps = 8  # Max observation window (max_tokens)
    
    # Variable observation lengths per sample (as shown in test results)
    observation_lengths = [8, 5, 3, 2]
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Horizon: {horizon}")
    print(f"  Max observation window: {max_obs_steps} (positions 0-{max_obs_steps-1})")
    print(f"  Observation lengths per sample: {observation_lengths}")
    
    # === STEP 1: Create trajectory data to denoise ===
    print(f"\n📊 STEP 1: Create trajectory data")
    print(f"-" * 35)
    
    # Sample trajectories (what the diffusion model will denoise)
    sample = torch.randn(batch_size, horizon, input_dim)
    
    # Diffusion timesteps (can be different per sample)
    timestep = torch.tensor([100, 200, 300, 400])
    
    # Target/goal conditioning (optional)
    target_cond = torch.randn(batch_size, target_dim)
    
    print(f"Created trajectory data:")
    print(f"  sample shape: {sample.shape}")
    print(f"  timesteps: {timestep.tolist()}")
    print(f"  target_cond shape: {target_cond.shape}")
    
    # === STEP 2: Create variable-length observation conditioning ===
    print(f"\n📊 STEP 2: Create variable-length observation conditioning")
    print(f"-" * 58)
    
    # Initialize padded tensors for batch
    max_tokens_in_batch = max(observation_lengths)  # 8
    global_cond = torch.zeros(batch_size, max_tokens_in_batch, global_cond_dim)
    global_mask = torch.zeros(batch_size, max_tokens_in_batch, dtype=torch.bool)
    temporal_positions = torch.zeros(batch_size, max_tokens_in_batch, dtype=torch.long)
    
    print(f"Initialized batch tensors:")
    print(f"  global_cond: {global_cond.shape}")
    print(f"  global_mask: {global_mask.shape}")
    print(f"  temporal_positions: {temporal_positions.shape}")
    
    # Fill each sample in the batch
    print(f"\nFilling batch with variable-length data:")
    
    for b in range(batch_size):
        num_obs_tokens = observation_lengths[b]
        
        # Create observation features for this sample (random for example)
        obs_features = torch.randn(num_obs_tokens, global_cond_dim)
        
        # Fill observation data
        global_cond[b, :num_obs_tokens] = obs_features
        
        # Set mask for valid tokens
        global_mask[b, :num_obs_tokens] = True
        
        # CRITICAL: Set relative temporal positions within max observation window
        # If max_obs_steps=8 and we have num_obs_tokens, use most recent positions
        relative_start = max_obs_steps - num_obs_tokens
        positions = torch.arange(relative_start, max_obs_steps)
        temporal_positions[b, :num_obs_tokens] = positions
        
        # Calculate what these positions mean in "steps ago"
        steps_ago = [max_obs_steps - pos for pos in positions.tolist()]
        
        print(f"  Sample {b} ({num_obs_tokens} tokens):")
        print(f"    Positions: {positions.tolist()}")
        print(f"    Steps ago: {steps_ago} (most recent observations)")
        print(f"    Mask: {global_mask[b].tolist()}")
    
    # === STEP 3: Verify temporal positioning matches test results ===
    print(f"\n🔍 STEP 3: Verify temporal positioning")
    print(f"-" * 38)
    
    expected_positions = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # Sample 0: 8 tokens
        [3, 4, 5, 6, 7],           # Sample 1: 5 tokens
        [5, 6, 7],                 # Sample 2: 3 tokens
        [6, 7]                     # Sample 3: 2 tokens
    ]
    
    print(f"Verification against expected positions:")
    for b in range(batch_size):
        num_tokens = observation_lengths[b]
        actual_positions = temporal_positions[b, :num_tokens].tolist()
        expected = expected_positions[b]
        
        match = actual_positions == expected
        status = "✅" if match else "❌"
        
        print(f"  Sample {b}: {actual_positions} {status}")
        if not match:
            print(f"    Expected: {expected}")
            raise AssertionError(f"Positions don't match for sample {b}")
    
    print(f"✅ All temporal positions match test results!")
    
    # === STEP 4: Use with AttentionConditionalUnet1D ===
    print(f"\n🤖 STEP 4: Use with AttentionConditionalUnet1D")
    print(f"-" * 43)
    
    # Create model
    model = AttentionConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        target_dim=target_dim,
        down_dims=[256, 512, 1024],
        max_global_tokens=max_obs_steps,
        num_attention_heads=8,
        attention_dropout=0.1
    )
    model.eval()
    
    print(f"Created AttentionConditionalUnet1D model")
    
    # Forward pass with variable-length conditioning
    with torch.no_grad():
        output = model(
            sample=sample,
            timestep=timestep,
            global_cond=global_cond,
            global_mask=global_mask,
            target_cond=target_cond,
            temporal_positions=temporal_positions
        )
    
    print(f"Forward pass results:")
    print(f"  Input shape: {sample.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  ✅ Successfully processed variable-length batch!")
    
    # === STEP 5: Show batch details ===
    print(f"\n📋 STEP 5: Complete batch details")
    print(f"-" * 34)
    
    print(f"Final batch structure:")
    print(f"  Trajectories to denoise: {sample.shape}")
    print(f"  Timesteps: {timestep.tolist()}")
    print(f"  Global conditioning: {global_cond.shape}")
    print(f"  Global mask: {global_mask.shape}")
    print(f"  Temporal positions: {temporal_positions.shape}")
    print(f"  Target conditioning: {target_cond.shape}")
    
    print(f"\nObservation window interpretation:")
    print(f"  Position 0 = {max_obs_steps} steps ago (oldest possible)")
    print(f"  Position 1 = {max_obs_steps-1} steps ago")
    print(f"  ...")
    print(f"  Position {max_obs_steps-1} = 1 step ago (most recent)")
    
    print(f"\nSample interpretations:")
    for b in range(batch_size):
        num_tokens = observation_lengths[b]
        valid_positions = temporal_positions[b, :num_tokens].tolist()
        steps_ago = [max_obs_steps - pos for pos in valid_positions]
        
        print(f"  Sample {b}: Positions {valid_positions} = {num_tokens} observations from {steps_ago} steps ago")
    
    return {
        'sample': sample,
        'timestep': timestep,
        'global_cond': global_cond,
        'global_mask': global_mask,
        'temporal_positions': temporal_positions,
        'target_cond': target_cond,
        'model': model
    }

def create_batch_from_observation_histories():
    """
    More realistic example: create batch from actual observation histories
    of different lengths, as would happen in practice.
    """
    
    print(f"\n🌟 REALISTIC EXAMPLE: From Observation Histories")
    print(f"=" * 60)
    
    # Simulate observation histories of different lengths (as collected during episodes)
    obs_histories = [
        torch.randn(20, 256),  # Episode had 20 observations
        torch.randn(15, 256),  # Episode had 15 observations  
        torch.randn(10, 256),  # Episode had 10 observations
        torch.randn(12, 256),  # Episode had 12 observations
    ]
    
    # How many observations to use from each history (variable curriculum)
    tokens_to_use = [8, 5, 3, 2]  # Could be determined by curriculum/random sampling
    
    max_obs_steps = 8
    batch_size = len(obs_histories)
    
    print(f"Observation histories:")
    for i, (history, tokens) in enumerate(zip(obs_histories, tokens_to_use)):
        print(f"  Episode {i}: {history.shape[0]} observations available, using {tokens} most recent")
    
    # Create batch following the same pattern
    global_cond = torch.zeros(batch_size, max_obs_steps, 256)
    global_mask = torch.zeros(batch_size, max_obs_steps, dtype=torch.bool)
    temporal_positions = torch.zeros(batch_size, max_obs_steps, dtype=torch.long)
    
    for b in range(batch_size):
        history = obs_histories[b]
        num_tokens = tokens_to_use[b]
        
        # Take most recent observations from history
        history_length = history.shape[0]
        start_idx = max(0, history_length - num_tokens)
        recent_obs = history[start_idx:start_idx + num_tokens]
        
        # Fill batch
        global_cond[b, :num_tokens] = recent_obs
        global_mask[b, :num_tokens] = True
        
        # Temporal positions (relative to observation window)
        relative_start = max_obs_steps - num_tokens
        temporal_positions[b, :num_tokens] = torch.arange(relative_start, max_obs_steps)
    
    print(f"\nCreated batch:")
    print(f"  global_cond: {global_cond.shape}")
    print(f"  global_mask: {global_mask.shape}")  
    print(f"  temporal_positions: {temporal_positions.shape}")
    
    print(f"\nTemporal positions:")
    for b in range(batch_size):
        num_tokens = tokens_to_use[b]
        positions = temporal_positions[b, :num_tokens].tolist()
        print(f"  Episode {b}: {positions} ({num_tokens} most recent observations)")
    
    return global_cond, global_mask, temporal_positions

def manual_batch_construction_step_by_step():
    """
    Step-by-step manual construction to show exact process.
    """
    
    print(f"\n🔨 MANUAL STEP-BY-STEP CONSTRUCTION")
    print(f"=" * 60)
    
    max_obs_steps = 8
    obs_dim = 256
    
    # Sample 0: 8 tokens → positions [0,1,2,3,4,5,6,7]
    print(f"Sample 0: 8 tokens")
    sample0_tokens = 8
    sample0_relative_start = max_obs_steps - sample0_tokens  # 8 - 8 = 0
    sample0_positions = list(range(sample0_relative_start, max_obs_steps))  # [0,1,2,3,4,5,6,7]
    print(f"  relative_start = {max_obs_steps} - {sample0_tokens} = {sample0_relative_start}")
    print(f"  positions = range({sample0_relative_start}, {max_obs_steps}) = {sample0_positions}")
    
    # Sample 1: 5 tokens → positions [3,4,5,6,7]
    print(f"\nSample 1: 5 tokens")
    sample1_tokens = 5
    sample1_relative_start = max_obs_steps - sample1_tokens  # 8 - 5 = 3
    sample1_positions = list(range(sample1_relative_start, max_obs_steps))  # [3,4,5,6,7]
    print(f"  relative_start = {max_obs_steps} - {sample1_tokens} = {sample1_relative_start}")
    print(f"  positions = range({sample1_relative_start}, {max_obs_steps}) = {sample1_positions}")
    
    # Sample 2: 3 tokens → positions [5,6,7] 
    print(f"\nSample 2: 3 tokens")
    sample2_tokens = 3
    sample2_relative_start = max_obs_steps - sample2_tokens  # 8 - 3 = 5
    sample2_positions = list(range(sample2_relative_start, max_obs_steps))  # [5,6,7]
    print(f"  relative_start = {max_obs_steps} - {sample2_tokens} = {sample2_relative_start}")
    print(f"  positions = range({sample2_relative_start}, {max_obs_steps}) = {sample2_positions}")
    
    # Sample 3: 2 tokens → positions [6,7]
    print(f"\nSample 3: 2 tokens")
    sample3_tokens = 2
    sample3_relative_start = max_obs_steps - sample3_tokens  # 8 - 2 = 6
    sample3_positions = list(range(sample3_relative_start, max_obs_steps))  # [6,7]
    print(f"  relative_start = {max_obs_steps} - {sample3_tokens} = {sample3_relative_start}")
    print(f"  positions = range({sample3_relative_start}, {max_obs_steps}) = {sample3_positions}")
    
    # Create the actual tensors
    print(f"\nConstructing tensors:")
    batch_size = 4
    max_tokens = max_obs_steps
    
    global_cond = torch.zeros(batch_size, max_tokens, obs_dim)
    global_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)
    temporal_positions = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    
    # Fill sample by sample
    samples_data = [
        (sample0_tokens, sample0_positions),
        (sample1_tokens, sample1_positions),
        (sample2_tokens, sample2_positions),
        (sample3_tokens, sample3_positions)
    ]
    
    for b, (num_tokens, positions) in enumerate(samples_data):
        # Fill with random observation data
        global_cond[b, :num_tokens] = torch.randn(num_tokens, obs_dim)
        
        # Set valid mask
        global_mask[b, :num_tokens] = True
        
        # Set temporal positions
        temporal_positions[b, :num_tokens] = torch.tensor(positions)
        
        print(f"  Sample {b}: filled {num_tokens} tokens at positions {positions}")
    
    print(f"\nFinal batch shapes:")
    print(f"  global_cond: {global_cond.shape}")
    print(f"  global_mask: {global_mask.shape}")
    print(f"  temporal_positions: {temporal_positions.shape}")
    
    print(f"\nTemporal positions tensor:")
    print(f"  {temporal_positions}")
    
    print(f"\nGlobal mask tensor:")
    print(f"  {global_mask}")
    
    return global_cond, global_mask, temporal_positions

if __name__ == "__main__":
    # Run the main example
    batch_data = create_variable_length_batch_example()
    
    # Show realistic example
    create_batch_from_observation_histories()
    
    # Show step-by-step manual construction
    manual_batch_construction_step_by_step()
    
    print(f"\n🎉 ALL EXAMPLES COMPLETED!")
    print(f"You now have complete code examples for creating variable-length batches!")