import pytest
import sys
import os

# Ensure we can import from src
sys.path.append(os.getcwd())

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from dm_toolkit.ai.agent.transformer_model import (
    DuelTransformerWithActionEmbedding,
    compute_loss_hierarchical,
    extend_action_types,
    encode_action_hierarchical
)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_initialization():
    model = DuelTransformerWithActionEmbedding(
        vocab_size=100,
        num_action_types=3,
        max_params_per_action=10,
        d_model=32,
        nhead=2,
        num_layers=2
    )
    assert model.d_model == 32
    assert model.action_type_embedding.num_embeddings == 3
    assert model.action_type_embedding.embedding_dim == 16  # 32 // 2

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_forward_pass():
    model = DuelTransformerWithActionEmbedding(
        vocab_size=100,
        num_action_types=3,
        max_params_per_action=10,
        d_model=32,
        nhead=2,
        num_layers=2
    )

    # Batch size 2, Sequence length 5
    x = torch.randint(0, 100, (2, 5))
    action_type_logits, param_logits, value = model(x)

    # Check shapes
    assert action_type_logits.shape == (2, 3) # [B, num_action_types]
    assert param_logits.shape == (2, 3, 10) # [B, num_action_types, max_params]
    assert value.shape == (2, 1)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_extend_action_types():
    model = DuelTransformerWithActionEmbedding(
        vocab_size=100,
        num_action_types=3,
        max_params_per_action=10,
        d_model=32
    )

    extend_action_types(model, 2) # Add 2 new types

    assert model.action_type_embedding.num_embeddings == 5
    assert model.action_type_head[-1].out_features == 5

    # Check forward pass with extended model
    x = torch.randint(0, 100, (2, 5))
    action_type_logits, param_logits, value = model(x)

    assert action_type_logits.shape == (2, 5)
    assert param_logits.shape == (2, 5, 10)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not available")
def test_compute_loss_hierarchical():
    # Dummy outputs
    B = 2
    num_types = 3
    max_params = 4

    action_type_logits = torch.randn(B, num_types)
    param_logits = torch.randn(B, num_types, max_params)
    value_pred = torch.randn(B, 1)

    model_output = (action_type_logits, param_logits, value_pred)

    # Targets
    target_actions = torch.tensor([[0, 0], [1, 2]]) # [Type, Param]
    target_values = torch.randn(B)

    loss = compute_loss_hierarchical(model_output, target_actions, target_values)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0 # Scalar

def test_encode_action_hierarchical():
    # Test PASS
    assert encode_action_hierarchical({'type': 'PASS'}) == [0, 0]

    # Test MANA_CHARGE
    assert encode_action_hierarchical({'type': 'MANA_CHARGE', 'slot_index': 5}) == [1, 5]

    # Test PLAY_FROM_ZONE
    assert encode_action_hierarchical({'type': 'PLAY_FROM_ZONE', 'slot_index': 3}) == [2, 3]

    # Test Legacy PLAY_CARD -> PLAY_FROM_ZONE
    assert encode_action_hierarchical({'type': 'PLAY_CARD', 'slot_index': 1}) == [2, 1]

    # Test Enum support (mocking enum)
    class ActionTypeEnum:
        def __init__(self, name):
            self.name = name

    assert encode_action_hierarchical({'type': ActionTypeEnum('PASS')}) == [0, 0]

    # Test object input
    class ActionObj:
        def __init__(self, t, s=0):
            self.type = t
            self.slot_index = s

    assert encode_action_hierarchical(ActionObj('MANA_CHARGE', 2)) == [1, 2]
