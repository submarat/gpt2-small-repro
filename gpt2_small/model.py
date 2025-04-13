# GPT-2 Small Model Implementation
# Clean up imports and setup
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import circuitsvis as cv
import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt

from IPython.display import display
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers import PreTrainedTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

# Set device based on availability
device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

MAIN = __name__ == '__main__'

# Load reference GPT-2 model for benchmarking and testing
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
)

# ======================
# MODEL DEFINITION
# ======================

@dataclass
class TransformerConfig:
    """Configuration for the transformer model."""
    d_model: int = 768
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
    vocab: int = 50257
    seq_len: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, config: TransformerConfig):
        """
        Args:
            - config - TransformerConfig object containing hyperparameters
        """
        super().__init__()

        self.seq_len = config.seq_len
        self.d_head = config.d_head
        
        # Query projection matrix
        self.W_Q = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head, device=device))
        nn.init.normal_(self.W_Q, std=config.init_range)
        # Key projection matrix
        self.W_K = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head, device=device))
        nn.init.normal_(self.W_K, std=config.init_range)
        # Value projection matrix
        self.W_V = nn.Parameter(t.empty(config.n_heads, config.d_model, config.d_head, device=device))
        nn.init.normal_(self.W_V, std=config.init_range)
        # Output projection matrix to obtain final values
        self.W_O = nn.Parameter(t.empty(config.n_heads, config.d_head, config.d_model, device=device))
        nn.init.normal_(self.W_O, std=config.init_range)
        # Biases
        self.b_Q = nn.Parameter(t.zeros(config.n_heads, config.d_head, device=device))
        self.b_K = nn.Parameter(t.zeros(config.n_heads, config.d_head, device=device))
        self.b_V = nn.Parameter(t.zeros(config.n_heads, config.d_head, device=device))
        self.b_O = nn.Parameter(t.zeros(config.d_model, device=device))
        self.register_buffer("IGNORE", t.tensor(float("-inf"), device=device, dtype=t.float32))

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> tuple[Float[Tensor, "batch n_heads seq seq"], Float[Tensor, "batch seq d_model"]]:
        """
        Forward pass for the attention mechanism.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            
        Returns:
            Tuple of (attention_probs, output)
            - attention_probs: Tensor of shape (batch, n_heads, seq, seq)
            - output: Tensor of shape (batch, seq, d_model)
        """
        seq_len = x.shape[-2]
        d_head = self.d_head

        causal_mask = t.triu(t.ones(seq_len, seq_len, dtype=bool, device=device), diagonal=1)

        # Project input into Q, K, V using einops
        queries = einops.einsum(x, self.W_Q, 'batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head') + self.b_Q
        keys = einops.einsum(x, self.W_K, 'batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head') + self.b_K
        values = einops.einsum(x, self.W_V, 'batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head') + self.b_V

        # Calculate attention scores and apply scaling
        attn_scores = einops.einsum(
            queries, keys, 
            'batch seq_q n_heads d_head, batch seq_k n_heads d_head -> batch n_heads seq_q seq_k'
        ) / (self.d_head ** 0.5)

        # Create causal mask and apply it
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax to get attention probabilities
        attn_probs = t.softmax(attn_scores, dim=-1)

        # Calculate weighted average of the values
        outputs = einops.einsum(
            attn_probs, values,
            'batch n_heads seq_q seq_k, batch seq_k n_heads d_head -> batch seq_q n_heads d_head'
        )

        # Project back to d_model dimension
        out = einops.einsum(
            outputs, self.W_O,
            'batch seq n_heads d_head, n_heads d_head d_model -> batch seq d_model'
        ) + self.b_O

        return attn_probs, out

class LayerNorm(nn.Module):
    """Layer normalization module."""
    
    def __init__(self, config: TransformerConfig):
        """
        Args:
            - config - TransformerConfig object containing model configuration
        """
        super().__init__()
        self.w = nn.Parameter(t.ones(config.d_model, device=device))
        self.b = nn.Parameter(t.zeros(config.d_model, device=device))
        self.eps = config.layer_norm_eps

    def forward(self, x: Float[Tensor, 'batch seq d_model']) -> Float[Tensor, 'batch seq d_model']:
        """
        Forward pass for layer normalization.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            
        Returns:
            Normalized tensor of shape (batch, seq, d_model)
        """
        # Normalize to mean 0, variance 1
        means = x.mean(dim=(-1), keepdim=True)
        variances = x.var(dim=(-1), keepdim=True, unbiased=False)
        x = (x - means)/(variances + self.eps)**0.5
        
        # Scale and translate
        x = x * self.w
        x = x + self.b
        return x

class MLP(nn.Module):
    """Multi-layer perceptron module."""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
    
    def forward(self, x: Float[Tensor, 'batch seq d_model']) -> Float[Tensor, 'batch seq d_model']:
        """
        Forward pass for the MLP.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            
        Returns:
            Output tensor of shape (batch, seq, d_model)
        """
        pre = einops.einsum(
            x, self.W_in,
            "batch position d_model, d_model d_mlp -> batch position d_mlp", 
        ) + self.b_in
        post = gelu_new(pre)
        mlp_out = einops.einsum(
            post, self.W_out,
            "batch position d_mlp, d_mlp d_model -> batch position d_model", 
        ) + self.b_out
        return mlp_out

class TransformerBlock(nn.Module):
    """
    TransformerBlock is a module that wraps MLP and Attention.
    It presents a single layer in Transfomer decoder which is
    repeated several times.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = LayerNorm(config)
        self.attn = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)
    
    def forward(self, x: Float[Tensor, 'batch seq_len d_model']) -> Float[Tensor, 'batch seq_len d_model']:
        """
        Forward pass for the transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        x_norm = self.ln1(x)
        _, attn_out = self.attn(x_norm)
        x1 = attn_out + x
        x1_norm = self.ln2(x1)
        mlp_out = self.mlp(x1_norm)
        x2 = mlp_out + x1
        return x2

class Embedding(nn.Module):
    """
    Embedding is simply a linear projection of the input sequence
    after tokenization.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(config.vocab, config.d_model, device=device))
        nn.init.normal_(self.W_E, std=config.init_range)

    def forward(self, x: Int[Tensor, 'batch seq_len']) -> Float[Tensor, 'batch seq_len d_model']:
        """
        Forward pass for the embedding.
        
        Args:
            x: Input tensor of shape (batch, seq_len) with token indices
            
        Returns:
            Embedded tensor of shape (batch, seq_len, d_model)
        """
        return self.W_E[x]

class PositionalEmbedding(nn.Module):
    """Positional embedding to provide position information."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.W_pos = nn.Parameter(
            t.empty((config.seq_len, config.d_model), device=device)
        )
        nn.init.normal_(self.W_pos)
    
    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        """
        Forward pass for the positional embedding.
        
        Args:
            x: Input tensor of shape (batch, seq_len) - only used for shape
            
        Returns:
            Positional embedding tensor of shape (batch, seq_len, d_model)
        """
        # Get the positions up to the current sequence length
        batch, seq_len = x.shape
        pos = self.W_pos[:seq_len, :]
        return einops.repeat(pos, 'seq d_model -> batch seq d_model', batch=batch)

class Unembedding(nn.Module):
    """
    The final unembedding layer in the GPT-style Transformer
    unembeds the residual stream vectors for each position
    returning logits over the entire vocabulary that can be used
    for sampling autoregressively.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.W_U = nn.Parameter(t.empty((config.d_model, config.vocab), device=device))
        nn.init.normal_(self.W_U, std=config.init_range)

        self.b_U = nn.Parameter(t.zeros((config.vocab,), device=device, requires_grad=False))
    
    def forward(self, x: Float[Tensor, 'batch seq d_model']) -> Float[Tensor, 'batch seq vocab']:
        """
        Forward pass for the unembedding layer.
        
        Args:
            x: Input tensor of shape (batch, seq, d_model)
            
        Returns:
            Logits tensor of shape (batch, seq, vocab)
        """
        return einops.einsum(x, self.W_U, 'batch seq d_model, d_model vocab -> batch seq vocab') + self.b_U

class Transformer(nn.Module):
    """
    Transformer is a stack of TransformerBlocks.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        # Store config as an attribute so the generation function can access it
        self.config = config
        
        self.embed = Embedding(config)
        self.pos_embed = PositionalEmbedding(config)

        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])

        self.ln_final = LayerNorm(config)
        self.unembed = Unembedding(config)
    
    def forward(self, tokens: Int[Tensor, 'batch seq']) -> Float[Tensor, 'batch seq vocab']:
        """
        Forward pass for the transformer.
        
        Args:
            tokens: Input tensor of shape (batch, seq) with token indices
            
        Returns:
            Logits tensor of shape (batch, seq, vocab)
        """
        x_embed = self.embed(tokens)
        x_pos = self.pos_embed(tokens)
        x = x_embed + x_pos

        x = self.blocks(x)
        x_norm = self.ln_final(x)
        logits = self.unembed(x_norm)
        return logits

# ======================
# UTILITY FUNCTIONS
# ======================

def generate_dataset(seq_len = 512):
    """Generate a simple dataset for testing."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models process text by breaking it into tokens.",
        "Neural networks have transformed natural language processing.",
        "Deep learning techniques are widely used in computer vision.",
        "Reinforcement learning is a type of machine learning.",
        "Natural language understanding is a challenging task.",
        "Transfer learning helps in improving model performance.",
        "Convolutional neural networks are effective for image recognition.",
        "Generative adversarial networks can create realistic images.",
        "Attention mechanisms improve the performance of neural networks."
        "Attention mechanisms improve the performance of neural networks."
        "Attention mechanisms improve the performance of neural networks."
    ]

    # Instantiate the tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the sample texts with padding_side='right' and pad to max_length
    tokenized_texts = [
        tokenizer(
            text, 
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=seq_len,
            padding_side='right'
        ) 
        for text in sample_texts
    ]

    return tokenized_texts

def load_wikitext_dataset(config: TransformerConfig, batch_size: int = 64):
    """Load the WikiText dataset for training."""
    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=config.seq_len, column_name="text", add_bos_token=True, num_proc=4)

    # Create DataLoader for batching
    tensor_dataset = t.utils.data.TensorDataset(tokenized_dataset['tokens'])
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def load_pile_10k_dataset(batch_size, max_length):
    """Load the Pile-10k dataset for training."""
    dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns("meta")
    tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=max_length, column_name="text", add_bos_token=True, num_proc=4)
    tokenized_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=max_length, column_name="text", add_bos_token=True, num_proc=4)

    dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
    train_loader = DataLoader(dataset_dict["train"], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset_dict["test"], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader

# ======================
# TRAINING FUNCTIONS
# ======================

def train(model, config):
    """Train the model using the specified dataset."""
    
    # Hyperparameters
    lr = 1e-3
    epochs = 50
    batch_size = 64
    weight_decay = 1e-2
    max_iter_per_epoch = 50 

    wandb.init(project="transformer_training", config={
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "max_iter_per_epoch": max_iter_per_epoch
    })

    # Get dataloader
    dataloader = load_wikitext_dataset(config, batch_size=batch_size)

    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    table = wandb.Table(columns=["input", "next_tokens"])
    step = 0
    for epoch in tqdm(range(epochs)):
        loss = t.ones(1)
        iterations = 0
        for batch_idx, batch_tokens in enumerate(tqdm(dataloader, desc="Batch Progress")):
            step += 1
            tqdm.write(f"Batch {batch_idx + 1}/{len(dataloader)}")
            # batch_tokens will be shape [batch_size, seq_len]
            batch_tokens = batch_tokens[0].to(device)  # Move to device

            # Calculate loss
            logits = model(batch_tokens)
            log_probs = logits.log_softmax(dim=-1)
            log_probs_for_tokens = log_probs[:, :-1]\
                .gather(dim=-1, index=batch_tokens[:, 1:].unsqueeze(-1)).unsqueeze(-1)
            loss = -log_probs_for_tokens.mean()

            wandb.log({"loss": loss}, step=step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iterations > max_iter_per_epoch:
                break
            iterations += 1
        
        # Log the last loss for the epoc
        wandb.log({"epoch": epoch, "loss": loss.item()})

        # Decode the final batch_tokens and logits by greedy sampling using reference_gpt2.tokenizer.decode
        decoded_batch_tokens = reference_gpt2.tokenizer.decode(batch_tokens[0])
        next_tokens = reference_gpt2.tokenizer.decode(logits[0,:].argmax(dim=-1))

        table.add_data(decoded_batch_tokens, next_tokens)

    wandb.log({"examples": table})
    
    wandb.finish()

# ======================
# TEXT GENERATION FUNCTIONS
# ======================

def generate():
    """Generate text from the model."""
    with t.no_grad():
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        sample_text = "Grab the"
        # Tokenize the sample text
        tokens = tokenizer(
                sample_text, 
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=config.seq_len,
                padding_side='right',
            )['input_ids']
        tokens = tokens.to(device)

        seq_len = (tokens[0] == tokenizer.pad_token_id).nonzero()[0].item()

        for n in range(seq_len, config.seq_len):
            logits = model(tokens)
            top_5_tokens = logits[0, n-1].topk(5).indices
            prefix = tokenizer.decode(tokens[0, :n])
            print(f"{prefix}")
            print(f": {[tokenizer.decode([token_id]) for token_id in top_5_tokens]}")
            next_token = top_5_tokens[0]
            tokens[0, n] = next_token # replace pad token with new prediction

        output_text = tokenizer.decode(token_ids=tokens[0])
        print("Sample output: ", output_text)

def predict(input_text):
    """Predict the next tokens for a given input text."""
    with t.no_grad():
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Tokenize the input text with padding_side='right' and pad to max_length
        tokens = tokenizer(
                input_text, 
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=config.seq_len,
                padding_side='right',
            )['input_ids']
        tokens = tokens.to(device)

        seq_len = (tokens[0] == tokenizer.pad_token_id).nonzero()[0].item()

        logits = model(tokens)
        print(f"{logits.shape}")
        print(f": {tokenizer.decode(logits[0].argmax(dim=-1))}")
