#!/usr/bin/env python
"""
Text generation demonstration for the gpt2_small package.

This script demonstrates how to generate text using models from the gpt2_small package.
It shows different generation strategies and parameters.
"""

import torch as t
from transformer_lens import HookedTransformer

from gpt2_small import (
    TransformerConfig,
    Transformer,
    device,
    generate
)

def main():
    """Run generation demonstrations."""
    print(f"Running on device: {device}")
    print("=" * 50)
    print("GPT-2 SMALL TEXT GENERATION DEMONSTRATION")
    print("=" * 50)
    
    # ===== Option 1: Generate with a randomly initialized small model =====
    print("\n1. Generation with a small randomly initialized model:\n")
    
    # Create a tiny model for fast demonstration
    config = TransformerConfig(
        d_model=256,
        n_heads=4,
        n_layers=2,
        d_mlp=512,
        seq_len=128,
        vocab=50257  # Standard GPT-2 vocabulary size
    )
    
    model = Transformer(config).to(device)
    # Store config as an attribute so the generation function can access it
    model.config = config
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    prompts = [
        "The quick brown fox",
        "Once upon a time",
        "The meaning of life is",
        "Artificial intelligence will"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: \"{prompt}\"")
        generated = generate(model, prompt=prompt, max_tokens=20)
        print(f"Generated: \"{generated}\"")
    
    # ===== Option 2: Generate with different parameters =====
    print("\n\n2. Generation with different parameters:\n")
    
    prompt = "The future of technology is"
    print(f"Prompt: \"{prompt}\"")
    
    print("\nLow temperature (more deterministic):")
    generated = generate(model, prompt=prompt, max_tokens=20, temperature=0.5)
    print(f"Generated (temp=0.5): \"{generated}\"")
    
    print("\nHigh temperature (more random):")
    generated = generate(model, prompt=prompt, max_tokens=20, temperature=1.5)
    print(f"Generated (temp=1.5): \"{generated}\"")
    
    print("\nWith top-k sampling (k=5):")
    generated = generate(model, prompt=prompt, max_tokens=20, top_k=5)
    print(f"Generated (top_k=5): \"{generated}\"")
    
    # ===== Option 3: Load a pre-trained model if available =====
    try:
        print("\n\n3. Generation with pre-trained weights (if available):\n")
        
        # Try to load the HookedTransformer GPT-2 model
        reference_gpt2 = HookedTransformer.from_pretrained(
            "gpt2-small",
            fold_ln=False,
            center_unembed=False,
            center_writing_weights=False,
            device=device
        )
        
        # Create a matching Transformer and copy weights
        full_config = TransformerConfig()  # Default is GPT-2 small
        pretrained_model = Transformer(full_config).to(device)
        # Store config as an attribute
        pretrained_model.config = full_config
        
        # Copy weights (this would need to be adjusted based on architecture differences)
        print("Transferring weights from pre-trained model...")
        pretrained_model.load_state_dict(reference_gpt2.state_dict(), strict=False)
        
        # Generate with the pre-trained model
        for prompt in prompts[:2]:  # Just use first two prompts
            print(f"\nPrompt: \"{prompt}\"")
            generated = generate(pretrained_model, prompt=prompt, max_tokens=30)
            print(f"Generated: \"{generated}\"")
            
    except Exception as e:
        print(f"\nCould not load pre-trained model: {e}")
        print("Skipping pre-trained model demonstration")
        
    print("\n" + "=" * 50)
    print("GENERATION DEMONSTRATION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main() 