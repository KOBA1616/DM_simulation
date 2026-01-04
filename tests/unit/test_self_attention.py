
import sys
import os
import pytest
import math

# Add bin directory to path to import dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

def test_tensor_basics():
    rows = 2
    cols = 3
    t = dm_ai_module.Tensor2D(rows, cols)

    assert t.rows == rows
    assert t.cols == cols
    assert len(t.data) == rows * cols

    # Check default initialization (0.0)
    for v in t.data:
        assert v == 0.0

def test_self_attention_structure():
    embed_dim = 4
    num_heads = 2
    sa = dm_ai_module.SelfAttention(embed_dim, num_heads)

    # Just running initialization to check for crashes
    sa.initialize_weights()

def test_self_attention_forward():
    embed_dim = 4
    num_heads = 2
    seq_len = 3

    sa = dm_ai_module.SelfAttention(embed_dim, num_heads)
    sa.initialize_weights()

    input_tensor = dm_ai_module.Tensor2D(seq_len, embed_dim)
    # Fill with some dummy data
    for i in range(len(input_tensor.data)):
        input_tensor.data[i] = float(i) * 0.1

    mask = [True, True, True] # All valid

    output = sa.forward(input_tensor, mask)

    assert output.rows == seq_len
    assert output.cols == embed_dim
    assert len(output.data) == seq_len * embed_dim

def test_self_attention_masking():
    embed_dim = 4
    num_heads = 2
    seq_len = 3

    sa = dm_ai_module.SelfAttention(embed_dim, num_heads)
    sa.initialize_weights()

    input_tensor = dm_ai_module.Tensor2D(seq_len, embed_dim)
    mask = [True, False, False] # Only first element valid

    # This shouldn't crash
    output = sa.forward(input_tensor, mask)

    assert output.rows == seq_len
    assert output.cols == embed_dim
