#!/usr/bin/env python3
"""Debug target conditioning parameter differences."""

import torch
from diffusion_policy.model.diffusion.attention_conditional_unet1d import AttentionConditionalUnet1D

def debug_target_params():
    # Same config as test
    batch_size = 2
    input_dim = 2
    global_cond_dim = 128
    target_dim = 2
    max_global_tokens = 4
    
    print("Creating models...")
    
    model_with_target = AttentionConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        target_dim=target_dim,
        max_global_tokens=max_global_tokens,
        use_target_conditioning=True,
        down_dims=[128, 256]
    )
    
    model_without_target = AttentionConditionalUnet1D(
        input_dim=input_dim,
        global_cond_dim=global_cond_dim,
        target_dim=target_dim,
        max_global_tokens=max_global_tokens,
        use_target_conditioning=False,
        down_dims=[128, 256]
    )
    
    print(f"unified_token_dim: {model_with_target.unified_token_dim}")
    print(f"target_dim: {target_dim}")
    print(f"Expected target projection params: {target_dim * model_with_target.unified_token_dim}")
    
    print(f"\nTarget projection layers:")
    print(f"  With target: {model_with_target.target_to_token}")
    print(f"  Without target: {model_without_target.target_to_token}")
    
    if model_with_target.target_to_token is not None:
        target_params = sum(p.numel() for p in model_with_target.target_to_token.parameters())
        print(f"  Target projection params: {target_params}")
    
    # Compare all parameter names to see what's different
    params_with = dict(model_with_target.named_parameters())
    params_without = dict(model_without_target.named_parameters())
    
    print(f"\nParameter differences:")
    for name in params_with:
        if name not in params_without:
            print(f"  Only in WITH target: {name} - {params_with[name].numel()} params")
    
    for name in params_without:
        if name not in params_with:
            print(f"  Only in WITHOUT target: {name} - {params_without[name].numel()} params")
    
    # Check if any existing parameters changed size
    for name in params_with:
        if name in params_without:
            if params_with[name].shape != params_without[name].shape:
                print(f"  Shape difference in {name}:")
                print(f"    With target: {params_with[name].shape} ({params_with[name].numel()} params)")
                print(f"    Without target: {params_without[name].shape} ({params_without[name].numel()} params)")

if __name__ == "__main__":
    debug_target_params()