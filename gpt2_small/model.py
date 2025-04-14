# GPT-2 Small Model Implementation
from dataclasses import dataclass

import einops
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float, Int
from torch import Tensor

# Set device based on availability
device = t.device('mps' if t.backends.mps.is_available() else 'cuda' if t.cuda.is_available() else 'cpu')

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
    
    def __init__(self, cfg: TransformerConfig):
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
        post = F.gelu(pre)
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
