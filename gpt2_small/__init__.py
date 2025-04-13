"""
GPT-2 Small implementation package.

A clean PyTorch implementation of a GPT-2 style transformer model.
"""

# Version information
__version__ = "0.1.0"

# Import important components from model.py
from .model import (
    TransformerConfig,
    MultiHeadAttention,
    LayerNorm,
    MLP,
    TransformerBlock,
    Embedding,
    PositionalEmbedding,
    Unembedding,
    Transformer,
    device,
)

# Import training utilities
from .training import (
    generate_dataset,
    load_wikitext_dataset,
    load_pile_10k_dataset,
    train,
)

# Import generation utilities
from .generation import (
    generate,
    predict,
) 