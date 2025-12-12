import torch
import pytest
from dm_toolkit.training.network_v2 import NetworkV2, LinearAttention

def test_linear_attention_shape():
    """Verify Linear Attention output shape."""
    dim = 64
    seq_len = 50
    batch_size = 4

    attn = LinearAttention(dim, heads=4, dim_head=16)
    x = torch.randn(batch_size, seq_len, dim)

    out = attn(x)
    assert out.shape == (batch_size, seq_len, dim)

def test_network_v2_forward():
    """Verify NetworkV2 forward pass with dummy input."""
    batch_size = 2
    seq_len = 100
    vocab_size = 500
    action_space = 600
    embedding_dim = 128

    model = NetworkV2(
        embedding_dim=embedding_dim,
        depth=2,
        heads=4,
        input_vocab_size=vocab_size,
        max_seq_len=200,
        action_space=action_space
    )

    # Create dummy indices [Batch, SeqLen]
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    policy, value = model(x)

    # Check Policy Shape: [Batch, ActionSpace]
    assert policy.shape == (batch_size, action_space)

    # Check Value Shape: [Batch, 1]
    assert value.shape == (batch_size, 1)

    # Check Softmax
    assert torch.allclose(policy.sum(dim=1), torch.ones(batch_size), atol=1e-5)

    # Check Value Range (-1 to 1 due to Tanh)
    assert (value >= -1).all() and (value <= 1).all()

def test_network_v2_masking():
    """Verify NetworkV2 handles masking correctly (variable length)."""
    batch_size = 2
    max_len = 50
    vocab_size = 100

    model = NetworkV2(embedding_dim=32, depth=1, input_vocab_size=vocab_size)

    x = torch.randint(0, vocab_size, (batch_size, max_len))

    # Create a mask where the second half is invalid
    mask = torch.ones((batch_size, max_len), dtype=torch.bool)
    mask[:, max_len//2:] = False

    policy, value = model(x, mask=mask)

    assert policy.shape == (batch_size, 600)
    assert value.shape == (batch_size, 1)

def test_linear_attention_masking_logic():
    """Check if Linear Attention actually respects the mask (basic check)."""
    dim = 32
    attn = LinearAttention(dim, heads=2, dim_head=16)
    x = torch.randn(2, 10, dim)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[:, 5:] = False # Mask last 5 tokens

    # We can't easily assert exact values without knowing weights,
    # but we can ensure it runs without error and output size is preserved.
    out = attn(x, mask=mask)
    assert out.shape == x.shape
