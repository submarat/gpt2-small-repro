"""
Test module for the GPT-2 implementation.
Run this file to verify the model components are working correctly.
"""
import torch as t
import matplotlib.pyplot as plt
from torch import Tensor

# Import from the gpt2_small package
from gpt2_small import (
    TransformerConfig, 
    MultiHeadAttention, 
    LayerNorm, 
    MLP, 
    TransformerBlock, 
    Embedding,
    Unembedding,
    Transformer, 
    device
)

def test_mha():
    """Test the multi-head attention module."""
    config = TransformerConfig(seq_len=10, d_model=16, d_head=8, n_heads=2)
    mha = MultiHeadAttention(config)

    # Test 1: Check output shapes
    test_input = t.randn(2, config.seq_len, config.d_model, device=device)
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

def visualize_attention():
    """Visualize attention patterns."""
    config = TransformerConfig(seq_len=10, d_model=16, d_head=8, n_heads=2)
    mha = MultiHeadAttention(config)

    test_input = t.randn(2, config.seq_len, config.d_model, device=device)
    test_attn_probs, test_out = mha(test_input)

    plt.figure(figsize=(12, 4))
    for head in range(config.n_heads):
        plt.subplot(1, config.n_heads, head + 1)
        plt.imshow(test_attn_probs[0, head].detach().cpu())
        plt.title(f'Head {head}')
        plt.colorbar()
    plt.tight_layout()
    plt.show()

def test_layer_norm():
    """Test the layer normalization module."""
    batch, seq_len = 2, 3
    config = TransformerConfig(d_model=5)
    ln = LayerNorm(config)

    test_input = t.randn(batch, seq_len, config.d_model, device=device)
    test_output = ln(test_input)

    # Confirm that input and output shape match
    assert test_input.shape == test_output.shape

    # Compare to torch LayerNorm implementation
    torch_ln = t.nn.LayerNorm(config.d_model, device=device)
    torch_ln.weight = ln.w
    torch_ln.bias = ln.b

    expected_output = torch_ln(test_input)
    assert t.allclose(expected_output.cpu(), test_output.cpu())

    print('All tests passed!')

def test_mlp():
    """Test the MLP module."""
    config = TransformerConfig(seq_len=10, d_model=8, d_mlp=32)
    batch = 2

    mlp = MLP(config).to(device=device)
    test_input = t.randn(batch, config.seq_len, config.d_model, device=device)
    test_output = mlp(test_input)

    # Input/output both come from and return to residual stream
    assert test_input.shape == test_output.shape

    # Check for infinities and NaN
    assert not t.isnan(test_output).any(), "Output contains NaN values"
    assert not t.isinf(test_output).any(), "Output contains infinite values"

    print('All tests passed!')

def test_transformer_block():
    """Test the transformer block module."""
    config = TransformerConfig(seq_len=10, d_model=8, d_head=4, n_heads=2, d_mlp = 32)

    block = TransformerBlock(config).to(device=device)
    test_input = t.randn(2, config.seq_len, config.d_model, device=device)
    test_output = block(test_input)

    # Check shapes match
    assert test_input.shape == test_output.shape

    # Check for infinities and NaN
    assert not t.isnan(test_output).any(), "Output contains NaN values"
    assert not t.isinf(test_output).any(), "Output contains infinite values"

    print('All transformer block tests passed!')

def test_embedding():
    """Test the embedding module."""
    config = TransformerConfig(seq_len=10, d_model=8, vocab=1000)
    batch = 2

    embedding = Embedding(config).to(device=device)
    # Create random token indices between 0 and config.vocab-1
    test_input = t.randint(0, config.vocab, (batch, config.seq_len), device=device)
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
    manual_output = embedding.W_E[test_input]
    assert t.allclose(test_output, manual_output), "Embedding lookup doesn't match manual lookup"

    print('All embedding tests passed!')

def test_unembedding():
    """Test the unembedding module."""
    config = TransformerConfig(d_model=768, vocab=50257, seq_len=3)
    batch_size = 2

    # Create a random input tensor
    test_input = t.randn((batch_size, config.seq_len, config.d_model), device=device)

    # Initialize the Unembedding module
    unembedding = Unembedding(config).to(device=device)

    # Run the forward pass
    test_output = unembedding(test_input)

    # Check the shape of the output
    assert test_output.shape == (batch_size, config.seq_len, config.vocab), "Output shape is incorrect"

    # Check for infinities and NaN
    assert not t.isnan(test_output).any(), "Output contains NaN values"
    assert not t.isinf(test_output).any(), "Output contains infinite values"

    print('All unembedding tests passed!')

def test_transformer():
    """Test the full transformer model."""
    batch=2

    config = TransformerConfig(
        seq_len=10,
        d_model=5,
        d_head=8,
        n_heads=4,
        d_mlp=20,
        n_layers=2,
        vocab=100,
    )

    transformer = Transformer(config).to(device=device)
    test_input = t.randint(size=(batch, config.seq_len), high=config.vocab, device=device)

    test_output = transformer(test_input)
    assert test_output.shape == (batch, config.seq_len, config.vocab)
    assert test_output.dtype == t.float32

    print("All transformer tests passed!")

def run_all_tests():
    """Run all test functions."""
    test_mha()
    test_layer_norm()
    test_mlp()
    test_transformer_block()
    test_embedding()
    test_unembedding()
    test_transformer()
    print("\nAll tests completed successfully!")
    
if __name__ == "__main__":
    run_all_tests()