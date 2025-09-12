#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/abhinav/RLG/gcs-diffusion')

import torch
from diffusion_policy.model.diffusion.attention_conditional_unet1d import AttentionConditionalUnet1D

def test_basic_forward():
    """Test basic forward pass"""
    print("Testing basic forward pass...")
    
    # Create model
    model = AttentionConditionalUnet1D(
        input_dim=7,
        global_cond_dim=128, 
        target_dim=64,
        down_dims=[256, 512, 1024],
        max_global_tokens=5
    )
    model.eval()
    
    # Test data
    batch_size, seq_len = 4, 16
    sample = torch.randn(batch_size, seq_len, 7)
    timestep = torch.randint(0, 1000, (batch_size,))
    
    # Test 1: Only timestep conditioning
    print("  Test 1: Only timestep...")
    output1 = model(sample, timestep)
    print(f"    Input shape: {sample.shape}, Output shape: {output1.shape}")
    assert output1.shape == sample.shape
    
    # Test 2: With variable-length global conditioning
    print("  Test 2: With global conditioning...")
    global_cond = torch.randn(batch_size, 3, 128)
    global_mask = torch.ones(batch_size, 3, dtype=torch.bool)
    temporal_positions = torch.tensor([[13, 14, 15]] * batch_size)
    
    output2 = model(sample, timestep, 
                   global_cond=global_cond,
                   global_mask=global_mask,
                   temporal_positions=temporal_positions)
    print(f"    Output shape: {output2.shape}")
    assert output2.shape == sample.shape
    
    # Test 3: With target conditioning
    print("  Test 3: With target conditioning...")
    target_cond = torch.randn(batch_size, 64)
    output3 = model(sample, timestep,
                   global_cond=global_cond,
                   global_mask=global_mask, 
                   temporal_positions=temporal_positions,
                   target_cond=target_cond)
    print(f"    Output shape: {output3.shape}")
    assert output3.shape == sample.shape
    
    print("✅ Basic forward pass tests passed!")
    return True

def test_variable_lengths():
    """Test different conditioning lengths"""
    print("Testing variable conditioning lengths...")
    
    model = AttentionConditionalUnet1D(input_dim=4, global_cond_dim=64, max_global_tokens=8)
    model.eval()
    
    sample = torch.randn(2, 8, 4) 
    timestep = torch.randint(0, 1000, (2,))
    
    # Test different token counts
    for num_tokens in [1, 3, 5, 8]:
        print(f"  Testing {num_tokens} tokens...")
        global_cond = torch.randn(2, num_tokens, 64)
        global_mask = torch.ones(2, num_tokens, dtype=torch.bool)
        # Positions for most recent tokens
        positions = torch.arange(16-num_tokens, 16).unsqueeze(0).expand(2, -1)
        
        output = model(sample, timestep,
                      global_cond=global_cond,
                      global_mask=global_mask, 
                      temporal_positions=positions)
        
        assert output.shape == sample.shape
        print(f"    ✓ {num_tokens} tokens OK")
    
    print("✅ Variable length tests passed!")
    return True

def test_gradients():
    """Test gradient flow"""
    print("Testing gradient flow...")
    
    model = AttentionConditionalUnet1D(input_dim=4, global_cond_dim=32, target_dim=16)
    model.train()
    
    sample = torch.randn(2, 8, 4, requires_grad=True)
    timestep = torch.randint(0, 1000, (2,))
    global_cond = torch.randn(2, 2, 32, requires_grad=True)
    target_cond = torch.randn(2, 16, requires_grad=True)
    
    output = model(sample, timestep, 
                  global_cond=global_cond,
                  target_cond=target_cond)
    loss = output.mean()
    loss.backward()
    
    # Check gradients exist
    assert sample.grad is not None
    assert global_cond.grad is not None
    assert target_cond.grad is not None
    
    print("✅ Gradient flow tests passed!")
    return True

def main():
    """Run all tests"""
    print("🧪 Testing AttentionConditionalUnet1D...")
    
    try:
        test_basic_forward()
        test_variable_lengths() 
        test_gradients()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("The attention-based conditioning model is working correctly!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)