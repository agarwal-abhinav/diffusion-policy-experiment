#!/usr/bin/env python3
"""
Debug the positional encoding issues in AttentionConditionalUnet1D.
"""

import torch
from diffusion_policy.model.diffusion.attention_conditional_unet1d import AttentionConditionalUnet1D

def debug_positional_encoding_issues():
    """Debug the specific issues identified with token positioning."""
    
    print("🐛 DEBUGGING POSITIONAL ENCODING ISSUES")
    print("=" * 60)
    
    # Create model
    model = AttentionConditionalUnet1D(
        input_dim=2,
        global_cond_dim=256,
        target_dim=2,
        max_global_tokens=8,
        down_dims=[256, 512]  # Smaller for debugging
    )
    
    # Test inputs
    batch_size = 2
    horizon = 16
    sample = torch.randn(batch_size, horizon, 2)
    timestep = torch.tensor([100, 200])
    
    # Variable-length observations
    global_cond = torch.randn(batch_size, 3, 256)  # 3 observations
    global_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    temporal_positions = torch.tensor([[5, 6, 7], [5, 6, 7]])  # Recent observations
    target_cond = torch.randn(batch_size, 2)
    
    print(f"Input setup:")
    print(f"  global_cond shape: {global_cond.shape}")
    print(f"  temporal_positions: {temporal_positions}")
    print(f"  target_cond shape: {target_cond.shape}")
    
    # === ISSUE 1: Check token preparation ===
    print(f"\n🔍 ISSUE 1: Token Position Assignment")
    print(f"-" * 40)
    
    # Call _prepare_tokens to see what happens
    combined_tokens, combined_mask, combined_positions = model._prepare_tokens(
        timestep, global_cond, global_mask, target_cond, temporal_positions, batch_size
    )
    
    print(f"Combined token info:")
    print(f"  combined_tokens shape: {combined_tokens.shape}")
    print(f"  combined_mask shape: {combined_mask.shape}")
    print(f"  combined_positions shape: {combined_positions.shape}")
    
    print(f"\nToken sequence breakdown:")
    for b in range(batch_size):
        valid_mask = combined_mask[b]
        valid_positions = combined_positions[b][valid_mask]
        print(f"  Sample {b}:")
        print(f"    Valid mask: {valid_mask}")
        print(f"    Valid positions: {valid_positions}")
        print(f"    Position breakdown:")
        
        # Analyze each token type
        token_idx = 0
        # Timestep token (always first)
        print(f"      Token {token_idx}: Timestep (position={combined_positions[b, token_idx].item()})")
        token_idx += 1
        
        # Observation tokens
        num_obs = global_cond.shape[1]
        for i in range(num_obs):
            if token_idx < combined_tokens.shape[1] and combined_mask[b, token_idx]:
                print(f"      Token {token_idx}: Observation {i} (position={combined_positions[b, token_idx].item()})")
            token_idx += 1
        
        # Target token
        if token_idx < combined_tokens.shape[1] and combined_mask[b, token_idx]:
            print(f"      Token {token_idx}: Target (position={combined_positions[b, token_idx].item()})")
    
    # === ISSUE 2: Check max_temporal_position ===
    print(f"\n🔍 ISSUE 2: max_temporal_position Settings")
    print(f"-" * 43)
    
    # Check each attention module's max_temporal_position
    attention_modules = []
    
    # Check down modules
    for idx, (resnet, resnet2, downsample) in enumerate(model.down_modules):
        attention_modules.append((f"down_module_{idx}_resnet1", resnet.attention_conditioning))
        attention_modules.append((f"down_module_{idx}_resnet2", resnet2.attention_conditioning))
    
    # Check mid modules  
    for idx, mid_module in enumerate(model.mid_modules):
        attention_modules.append((f"mid_module_{idx}", mid_module.attention_conditioning))
    
    # Check up modules
    for idx, (resnet, resnet2, upsample) in enumerate(model.up_modules):
        attention_modules.append((f"up_module_{idx}_resnet1", resnet.attention_conditioning))
        attention_modules.append((f"up_module_{idx}_resnet2", resnet2.attention_conditioning))
    
    print(f"Attention module max_temporal_position settings:")
    for name, attention_module in attention_modules[:4]:  # Just check first few
        if hasattr(attention_module, 'temporal_pos_encoding'):
            max_pos = attention_module.temporal_pos_encoding.max_position
            print(f"  {name}: max_temporal_position = {max_pos}")
        else:
            print(f"  {name}: No temporal positional encoding")
    
    # === ISSUE 3: Check position clamping ===
    print(f"\n🔍 ISSUE 3: Position Clamping in TemporalPositionalEncoding")
    print(f"-" * 59)
    
    # Get one of the attention modules
    first_attention = model.down_modules[0][0].attention_conditioning
    if hasattr(first_attention, 'temporal_pos_encoding'):
        temp_pos_enc = first_attention.temporal_pos_encoding
        max_allowed = temp_pos_enc.max_position
        
        print(f"TemporalPositionalEncoding settings:")
        print(f"  max_position: {max_allowed}")
        print(f"  pe tensor shape: {temp_pos_enc.pe.shape}")
        
        # Test what happens with our positions
        test_positions = combined_positions[:1, :5]  # First sample, first 5 positions
        print(f"  Input positions: {test_positions}")
        
        # Check clamping
        clamped_positions = torch.clamp(test_positions, 0, max_allowed - 1)
        print(f"  Clamped positions: {clamped_positions}")
        
        # Identify problematic positions
        problematic = test_positions >= max_allowed
        if torch.any(problematic):
            print(f"  ⚠️  Positions >= {max_allowed}: {test_positions[problematic]}")
            print(f"  These will be clamped to {max_allowed - 1}!")
        else:
            print(f"  ✅ All positions within valid range")
    
    # === ISSUE 4: Try forward pass ===
    print(f"\n🔍 ISSUE 4: Forward Pass with Current Issues")
    print(f"-" * 44)
    
    try:
        with torch.no_grad():
            output = model(
                sample=sample,
                timestep=timestep,
                global_cond=global_cond,
                global_mask=global_mask,
                target_cond=target_cond,
                temporal_positions=temporal_positions
            )
        print(f"Forward pass succeeded:")
        print(f"  Output shape: {output.shape}")
        print(f"  ✅ No immediate errors, but positions may be incorrect")
    except Exception as e:
        print(f"Forward pass failed:")
        print(f"  ❌ Error: {e}")
    
    # === SUMMARY ===
    print(f"\n📋 SUMMARY OF ISSUES")
    print(f"-" * 25)
    
    issues_found = []
    
    # Check for position 999 (target token)
    if torch.any(combined_positions == 999):
        issues_found.append("Target token uses position 999 (way outside valid range)")
    
    # Check for position conflicts
    timestep_positions = combined_positions[:, 0]  # First token is always timestep
    if torch.any(timestep_positions != 0):
        issues_found.append("Timestep token not at position 0")
    
    # Check max_temporal_position
    if hasattr(first_attention, 'temporal_pos_encoding'):
        if first_attention.temporal_pos_encoding.max_position < 1000:
            issues_found.append(f"max_temporal_position={first_attention.temporal_pos_encoding.max_position} too small for position 999")
    
    print(f"Issues identified:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
    
    if not issues_found:
        print(f"  ✅ No major issues detected")
    
    return combined_tokens, combined_mask, combined_positions

if __name__ == "__main__":
    debug_positional_encoding_issues()