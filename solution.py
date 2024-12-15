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
@dataclass
class TransformerConfig:
    seq_len: int = 64
    d_model: int = 16
    d_head: int = 32
    n_heads: int = 8
    d_mlp: int = 16
    n_layers: int = 2
    vocab: int = 128
    device: str = 'cpu'

class MultiHeadAttention(nn.Module):

    def __init__(self, config: TransformerConfig):
        """
        Args:
            - config - TransformerConfig object containing hyperparameters
        """
        super().__init__()

        self.seq_len = config.seq_len
        self.d_head = config.d_head
        self.device = config.device
        
        self.causal_mask = t.triu(t.ones(config.seq_len, config.seq_len, dtype=bool, device=config.device), diagonal=1)

        # Query projection matrix
        self.W_q = nn.Parameter(t.randn(config.n_heads, config.d_model, config.d_head, device=config.device)) 
        # Key projection matrix
        self.W_k = nn.Parameter(t.randn(config.n_heads, config.d_model, config.d_head, device=config.device))
        # Value projection matrix
        self.W_v = nn.Parameter(t.randn(config.n_heads, config.d_model, config.d_head, device=config.device))
        # Output projection matrix to obtain final values
        self.W_o = nn.Parameter(t.randn(config.n_heads, config.d_head, config.d_model, device=config.device))

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
    config = TransformerConfig(seq_len=10, d_model=16, d_head=8, n_heads=2, device=device)
    mha = MultiHeadAttention(config)

    # Test 1: Check output shapes
    test_input = t.randn(2, config.seq_len, config.d_model, device=config.device)
    test_attn_probs, test_out = mha(test_input)

    expected_attn_shape = (2, config.n_heads, config.seq_len, config.seq_len)
    expected_out_shape = (2, config.seq_len, config.d_model)

    assert test_attn_probs.shape == expected_attn_shape, f"Attention probs shape {test_attn_probs.shape} != expected {expected_attn_shape}"
    assert test_out.shape == expected_out_shape, f"Output shape {test_out.shape} != expected {expected_out_shape}"

    # Test 2: Check attention probabilities sum to 1
    attn_probs_sum = test_attn_probs.sum(dim=-1)
    assert t.allclose(attn_probs_sum, t.ones_like(attn_probs_sum)), "Attention probabilities don't sum to 1"

    # Test 3: Verify causal attention mask
    for q_pos in range(config.seq_len):
        for k_pos in range(config.seq_len):
            if k_pos > q_pos:  # Future positions should have 0 attention
                assert t.allclose(test_attn_probs[..., q_pos, k_pos], t.zeros_like(test_attn_probs[..., q_pos, k_pos])), \
                    f"Non-causal attention at position q={q_pos}, k={k_pos}"

    print("All tests passed!")

test_mha()

# %%
# Visualize attention probabilities
config = TransformerConfig(seq_len=10, d_model=16, d_head=8, n_heads=2, device=device)
mha = MultiHeadAttention(config)

test_input = t.randn(2, config.seq_len, config.d_model, device=config.device)
test_attn_probs, test_out = mha(test_input)

plt.figure(figsize=(12, 4))
for head in range(config.n_heads):
    plt.subplot(1, config.n_heads, head + 1)
    plt.imshow(test_attn_probs[0, head].detach().cpu())
    plt.title(f'Head {head}')
    plt.colorbar()
plt.tight_layout()
plt.show()

# %%
# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Args:
            - config - TransformerConfig object containing model configuration
        """
        super().__init__()
        self.scale = nn.Parameter(t.ones(config.d_model, device=config.device))
        self.shift = nn.Parameter(t.zeros(config.d_model, device=config.device))
        self.eps = 1e-5

    def forward(self, x: Float[Tensor, 'batch seq d_model']) -> Float[Tensor, 'batch seq d_model']:
        # Normalize to mean 0, variance 1
        means = x.mean(dim=(-1), keepdim=True)
        variances = x.var(dim=(-1), keepdim=True, unbiased=False)
        x = (x - means)/(variances + self.eps)**0.5
        
        # Scale and translate
        x = x * self.scale
        x = x + self.shift
        return x


def test_layer_norm():
    batch, seq_len = 2, 3
    config = TransformerConfig(d_model=5, device=device)
    ln = LayerNorm(config)

    test_input = t.randn(batch, seq_len, config.d_model, device=config.device)
    test_output = ln(test_input)

    # Confirm that input and output shape match
    assert test_input.shape == test_output.shape

    # Compare to torch LayerNorm implementation
    torch_ln = nn.LayerNorm(config.d_model, device=config.device)
    torch_ln.weights = ln.scale
    torch_ln.bias = ln.shift

    expected_output = torch_ln(test_input)
    assert t.allclose(expected_output.cpu(), test_output.cpu())

    print('All tests pass!')

test_layer_norm()

# %%
# MLP layer
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.l1 = nn.Linear(config.d_model, config.d_mlp, device=device)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(config.d_mlp, config.d_model, device=device)
    
    def forward(self, x: Float[Tensor, 'batch seq d_model']) -> Float[Tensor, 'batch seq d_model']:
        return self.l2(self.gelu(self.l1(x)))

def test_mlp():
    config = TransformerConfig(seq_len=10, d_model=8, d_mlp=32, device=device)
    batch = 2

    mlp = MLP(config)
    test_input = t.randn(batch, config.seq_len, config.d_model, device=config.device)
    test_output = mlp(test_input)

    # Input/output both come from and return to residual stream
    assert test_input.shape == test_output.shape

    # Check for infinities and NaN
    assert not t.isnan(test_output).any(), "Output contains NaN values"
    assert not t.isinf(test_output).any(), "Output contains infinite values"

    # Compare with torch implementation
    print('all tests passed!')

test_mlp()

# %%
class TransformerBlock(nn.Module):
    """
    TransformerBlock is a module that wraps MLP and Attention.
    It presents a single layer in Transfomer decoder which is
    repeated several times.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = LayerNorm(config)
        self.mha = MultiHeadAttention(config)
        self.ln2 = LayerNorm(config)
        self.mlp = MLP(config)
    
    def forward(self, x: Float[Tensor, 'batch seq_len d_model']) -> Float[Tensor, 'batch seq_len d_model']:
        x_norm = self.ln1(x)
        _, attn_out = self.mha(x_norm)
        x1 = attn_out + x
        x1_norm = self.ln2(x1)
        mlp_out = self.mlp(x1_norm)
        x2 = mlp_out + x1
        return x2

def test_transformer_block():
    config = TransformerConfig(seq_len=10, d_model=8, d_head=4, n_heads=2, d_mlp = 32, device=device)

    block = TransformerBlock(config)
    test_input = t.randn(2, config.seq_len, config.d_model, device=config.device)
    test_output = block(test_input)

    # Check shapes match
    assert test_input.shape == test_output.shape

    # Check for infinities and NaN
    assert not t.isnan(test_output).any(), "Output contains NaN values"
    assert not t.isinf(test_output).any(), "Output contains infinite values"

    print('all transformer block tests passed!')

test_transformer_block()

# %%
# Embedding
class Embedding(nn.Module):
    """
    Embedding is simply a linear projection of the input sequence
    after tokenization.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = nn.Parameter(t.randn(config.vocab, config.d_model, device=config.device))

    def forward(self, x: Int[Tensor, 'batch seq_len']) -> Float[Tensor, 'batch seq_len d_model']:
        return self.embed[x]


def test_embedding():
    config = TransformerConfig(seq_len=10, d_model=8, vocab=1000, device=device)
    batch = 2

    embedding = Embedding(config)
    # Create random token indices between 0 and config.vocab-1
    test_input = t.randint(0, config.vocab, (batch, config.seq_len), device=config.device)
    test_output = embedding(test_input)

    # Check output shape is correct
    expected_shape = (batch, config.seq_len, config.d_model)
    assert test_output.shape == expected_shape, f"Expected shape {expected_shape}, got {test_output.shape}"

    # Check output type is float
    assert test_output.dtype == t.float32, f"Expected dtype float32, got {test_output.dtype}"

    # Check for infinities and NaN
    assert not t.isnan(test_output).any(), "Output contains NaN values"
    assert not t.isinf(test_output).any(), "Output contains infinite values"

    # Check that the embedding actually uses the embedding matrix
    # by verifying output matches manual lookup
    manual_output = embedding.embed[test_input]
    assert t.allclose(test_output, manual_output), "Embedding lookup doesn't match manual lookup"

    print('all embedding tests passed!')

test_embedding()

# %%
class PositionalEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            t.empty((config.seq_len, config.d_model), device=config.device)
        )
        nn.init.normal_(self.positional_embedding)
    
    def forward(self, x: Int[Tensor, "batch seq_len"]) -> Float[Tensor, "batch seq_len d_model"]:
        # Get the positions up to the current sequence length
        batch, seq_len = x.shape
        pos = self.positional_embedding[:seq_len, :]
        return einops.repeat(pos, 'seq d_model -> batch seq d_model', batch=batch)

# %%
class Unembedding(nn.Module):
    """
    The final unembedding layer in the GPT-style Transformer
    unembeds the residual stream vectors for each position
    returning logits over the entire vocabulary that can be used
    for sampling autoregressively.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w_u = nn.Parameter(t.empty((config.d_model, config.vocab), device=config.device))
        nn.init.normal_(self.w_u)
        self.b_u = nn.Parameter(t.zeros((config.vocab,), device=config.device, requires_grad=False))
    
    def forward(self, x: Float[Tensor, 'batch seq d_model']) -> Float[Tensor, 'batch seq vocab']:
        """
        We return a distribution over the vocabulary for the next most
        likely token at each position.
        """
        return einops.einsum(x, self.w_u, 'batch seq d_model, d_model vocab -> batch seq vocab') + self.b_u
        
def test_unembedding():
    config = TransformerConfig(d_model=768, vocab=50257, seq_len=3, device=device)
    batch_size = 2

    # Create a random input tensor
    test_input = t.randn((batch_size, config.seq_len, config.d_model), device=config.device)

    # Initialize the Unembedding module
    unembedding = Unembedding(config)

    # Run the forward pass
    test_output = unembedding(test_input)

    # Check the shape of the output
    assert test_output.shape == (batch_size, config.seq_len, config.vocab), "Output shape is incorrect"

    # Check for infinities and NaN
    assert not t.isnan(test_output).any(), "Output contains NaN values"
    assert not t.isinf(test_output).any(), "Output contains infinite values"

    print('all unembedding tests passed!')

test_unembedding()

# %%
class Transformer(nn.Module):
    """
    Transformer is a stack of TransformerBlocks.
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.embed = Embedding(config)
        self.pos_embed = PositionalEmbedding(config)

        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])

        self.layer_norm = LayerNorm(config)
        self.unembed = Unembedding(config)
    
    def forward(self, tokens: Int[Tensor, 'batch seq']) -> Float[Tensor, 'batch seq vocab']:
        x_embed = self.embed(tokens)
        x_pos = self.pos_embed(tokens)
        x = x_embed + x_pos

        x = self.blocks(x)
        x_norm = self.layer_norm(x)
        logits = self.unembed(x_norm)
        return logits
    
    def loss(self, tokens: Int[Tensor, 'batch seq']) -> Float[Tensor, '']:
        logits = self.forward(tokens)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tokens.view(-1))
        return loss

# Question: where is the autoregressive bit that's used for training?
# Answer: given a list of tokens we'll get back a bunch of distributions (as logits) over all tokens
# You could sample autoregressively from Transformer at this point
# The loss function is the thing that takes a sequence of tokens and compares the response
# i.e. given tokens[:-1] compare predicted distribution with the true labels
# tokens[1:]

def test_transformer():
    batch=2

    config = TransformerConfig(
        seq_len=10,
        d_model=5,
        d_head=8,
        n_heads=4,
        d_mlp=20,
        n_layers=2,
        vocab=100,
        device=device
    )

    transformer = Transformer(config)
    test_input = t.randint(size=(batch, config.seq_len), high=config.vocab)

    test_output = transformer(test_input)
    assert test_output.shape == (batch, config.seq_len, config.vocab)
    assert test_output.dtype == t.float

    print("All tests passed!")

test_transformer()

# %%
def generate_dataset(seq_len = 512):
    # Generate some sample text to train the transformer on
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

tokenized_texts = generate_dataset()

batch_tokens = t.cat([t.tensor(tokens.input_ids.clone().detach()) for tokens in tokenized_texts], dim=0)

# %%
# Next we need to implement the training loop
# This should include the cross-entropy loss function
# and the optimizer.

config = TransformerConfig(
    seq_len=16,
    d_model=48,
    d_head=16,
    n_heads=8,
    d_mlp=768,
    n_layers=8,
    vocab=50257,
    device=device
)

tokenized_texts = generate_dataset(config.seq_len)

# Hyperparameters
lr = 1e-4
epochs = 100
batch_size = 512

model = Transformer(config)

# %%
from datasets import load_dataset

# Load dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize all texts at once
tokenized_dataset = tokenizer(
    dataset['text'],
    return_tensors='pt',
    padding='max_length',
    truncation=True,
    max_length=config.seq_len,
    padding_side='right'
)

# %%
# Create DataLoader for batching
tensor_dataset = t.utils.data.TensorDataset(tokenized_dataset['input_ids'])
dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

# %%
def train(model, config):

    optimizer = t.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epochs)):
        loss = t.ones(1)
        for batch_idx, batch_tokens in enumerate(tqdm(dataloader, desc="Batch Progress")):
            tqdm.write(f"Batch {batch_idx + 1}/{len(dataloader)}")
            # batch_tokens will be shape [batch_size, seq_len]
            batch_tokens = batch_tokens[0].to(device)  # Move to device
            
            # Calculate loss
            loss = model.loss(batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

train(model, config)

# %%
# Save the model weights to file
model_save_path = "model_weights.pth"
t.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")

# %%
def generate():
    with t.no_grad():
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        sample_text = "quick brown fox"
        # Tokenize the sample text

        # Tokenize the sample texts with padding_side='right' and pad to max_length
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
generate()

# %%
def predict(input_text):
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

predict("India has system, and has been the world's most populous democracy since")

# %%

# %%
