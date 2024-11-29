# %%
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
import wandb
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

device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

MAIN = __name__ == '__main__'

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device
)

# %%
sorted_vocab = sorted(reference_gpt2.tokenizer.vocab.items(), key=lambda x: x[1])
print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
print(sorted_vocab[-20:])
print(len(sorted_vocab))

# %%
lengths = dict.fromkeys(range(3, 8), "")
for tok, idx in sorted_vocab:
    if not lengths.get(len(tok), True):
        lengths[len(tok)] = tok

for length, tok in lengths.items():
    print(f"{length}: {tok}")

# %%
print(reference_gpt2.to_str_tokens("Ralph"))
print(reference_gpt2.to_str_tokens(" Ralph"))
print(reference_gpt2.to_str_tokens(" ralph"))
print(reference_gpt2.to_str_tokens("ralph"))
# %%

# %%
print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))
# %%
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))
# %%
logits, cache = reference_gpt2.run_with_cache(tokens, device=device)
print(logits.shape)
# %%
probs = logits.softmax(dim=-1)
print(probs.shape)
# %%
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))

# %%
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))

# %%
# Complete the next 10 tokens

length_of_completion = 10
seq = tokens
for n in range(length_of_completion):
    logits, cache = reference_gpt2.run_with_cache(seq, device=device)
    probs = logits.softmax(dim=-1)
    next_token = logits[0, -1].argmax(dim=-1)
    next_token = t.Tensor(next_token).reshape((1, 1))
    seq = t.cat([seq, t.Tensor(next_token)], dim=1)

completion = reference_gpt2.to_string(seq)
print(completion)

# %%[markdown]
## Plan for implementation

# Implement network modules
# - Attention head
#     - Q, K, V weights, causal mask
#     - Attention probs, attention values
# - LayerNorm
# - MLP
#     - Linear layer, activation (ReLU)
# - Transformer block - attention head, layer norm, MLP
# - Embedding
# - Positional embedding
# - Unembedding
# - Transformer with batch, vocab, seq, d_model, d_head, n_heads, d_mlp for each module
# - Forward function

# Train
# Data loader - some text corpus of sequenes
# Cross-entropy loss function
# Optimizer
# Training loop: forward, loss, zero_grad, optimizer.step

# Sampling

# %%
# Self attention
# In this section we will implement the attention head which consists of
# Q, K, V tensors which we will multiple to obtain
class MultiHeadAttention(nn.Module):

    def __init__(self, seq_len: int, d_model: int, d_head: int, n_heads: int, device):
        """
        Args:
            - seq_len - context length
            - d_model - residual stream dimensions
            - d_head - dimensions in each head
            - n_heads - number of heads per attention head
        """
        super().__init__()

        self.seq_len = seq_len
        self.d_head = d_head
        
        self.causal_mask = t.triu(t.ones(seq_len, seq_len, dtype=bool, device=device), diagonal=1)

        # Query projection matrix
        self.W_q = nn.Parameter(t.randn(n_heads, d_model, d_head, device=device)) 
        # Key projection matrix
        self.W_k = nn.Parameter(t.randn(n_heads, d_model, d_head, device=device))
        # Value projection matrix
        self.W_v = nn.Parameter(t.randn(n_heads, d_model, d_head, device=device))
        # Output projection matrix to obtain final values
        self.W_o = nn.Parameter(t.randn(n_heads, d_head, d_model, device=device))

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        seq_len = self.seq_len
        d_head = self.d_head
        causal_mask = self.causal_mask

        # Project input into Q, K, V using einops
        queries = einops.einsum(x, self.W_q, 'batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head')
        keys = einops.einsum(x, self.W_k, 'batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head')
        values = einops.einsum(x, self.W_v, 'batch seq d_model, n_heads d_model d_head -> batch seq n_heads d_head')

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
            outputs, self.W_o,
            'batch seq n_heads d_head, n_heads d_head d_model -> batch seq d_model'
        )

        return attn_probs, out


def test_mha():
    batch_size, seq_len, d_model, d_head, n_heads = 2, 10, 16, 8, 2
    mha = MultiHeadAttention(seq_len, d_model, d_head, n_heads, device)

    # Test 1: Check output shapes
    test_input = t.randn(batch_size, seq_len, d_model, device=device)
    test_attn_probs, test_out = mha(test_input)

    expected_attn_shape = (batch_size, n_heads, seq_len, seq_len)
    expected_out_shape = (batch_size, seq_len, d_model)

    assert test_attn_probs.shape == expected_attn_shape, f"Attention probs shape {test_attn_probs.shape} != expected {expected_attn_shape}"
    assert test_out.shape == expected_out_shape, f"Output shape {test_out.shape} != expected {expected_out_shape}"

    # Test 2: Check attention probabilities sum to 1
    attn_probs_sum = test_attn_probs.sum(dim=-1)
    assert t.allclose(attn_probs_sum, t.ones_like(attn_probs_sum)), "Attention probabilities don't sum to 1"

    # Test 3: Verify causal attention mask
    for q_pos in range(seq_len):
        for k_pos in range(seq_len):
            if k_pos > q_pos:  # Future positions should have 0 attention
                assert t.allclose(test_attn_probs[..., q_pos, k_pos], t.zeros_like(test_attn_probs[..., q_pos, k_pos])), \
                    f"Non-causal attention at position q={q_pos}, k={k_pos}"

    print("All tests passed!")

test_mha()

# %%
# Visualize attention probabilities
batch_size, seq_len, d_model, d_head, n_heads = 2, 10, 16, 8, 2
mha = MultiHeadAttention(seq_len, d_model, d_head, n_heads, device)

# Test 1: Check output shapes
test_input = t.randn(batch_size, seq_len, d_model, device=device)
test_attn_probs, test_out = mha(test_input)