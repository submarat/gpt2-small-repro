#!/usr/bin/env python
"""
Training demonstration for the gpt2_small package.

This script demonstrates how to train a model and periodically generate text
to show training progress.
"""

import os
import time
import torch as t
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast

from gpt2_small import (
    TransformerConfig,
    Transformer,
    device,
    generate,
    train
)

class TrainingDemo:
    """Class to demonstrate model training with periodic generation."""
    
    def __init__(self):
        """Initialize the demo."""
        self.config = TransformerConfig(
            d_model=256,          # Small model for demonstration
            n_heads=4,
            n_layers=2,
            d_mlp=512,
            seq_len=64,           # Shorter sequences for faster training
            vocab=50257           # Standard GPT-2 vocabulary size
        )
        
        self.model = Transformer(self.config).to(device)
        # Store config as an attribute so the generation function can access it
        self.model.config = self.config
        print(f"Created model with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Test prompts for generation during training
        self.prompts = [
            "The quick brown fox",
            "Once upon a time",
            "The meaning of life is"
        ]
    
    def run_demo(self):
        """Run the training demonstration."""
        print("=" * 50)
        print("GPT-2 SMALL TRAINING DEMONSTRATION")
        print("=" * 50)
        print(f"\nRunning on device: {device}")
        
        # Parameters for the demo
        epochs = 5
        lr = 3e-4
        
        # Create output directory for samples
        os.makedirs("training_samples", exist_ok=True)
        
        print("\nSTARTING TRAINING:")
        print("-" * 30)
        
        start_time = time.time()
        train(
            model=self.model,
            config=self.config,
            use_wandb=False,      # No WandB logging for the demo
            epochs=epochs,
            lr=lr,
            batch_size=4,         # Small batch size for demonstration
            max_iter_per_epoch=10, # Limit iterations for demonstration
            dataset='wikitext',     # Use generated sample dataset
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")

        # Show final generation using gpt2_small.generate()
        print("\nFINAL GENERATION (AFTER TRAINING):")
        for prompt in self.prompts:
            print(f"\nPrompt: {prompt}")
            # Follow method signature: def generate(model, prompt="Grab the", max_tokens=50, temperature=1.0, top_k=None):
            generated_text = generate(
                model=self.model,
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                top_k=50
            )
            print(generated_text)
            
def main():
    """Run the demo."""
    demo = TrainingDemo()
    
    demo.run_demo()

if __name__ == "__main__":
    main() 