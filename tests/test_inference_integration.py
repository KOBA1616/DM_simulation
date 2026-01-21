import pytest
import torch
import numpy as np
import os
from pathlib import Path
import dm_ai_module
from dm_ai_module import GameInstance, ActionType

# Import the model architecture
from dm_toolkit.ai.agent.transformer_model import DuelTransformer

def test_inference_integration():
    """
    Verifies that a trained model can be loaded and used to generate an action
    from a GameInstance state.
    """

    # 1. Setup Model (Mock Training)
    # Create a small instance of the model
    vocab_size = 1000
    action_dim = 600
    model = DuelTransformer(
        vocab_size=vocab_size,
        action_dim=action_dim,
        d_model=64, # Small for test
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        max_len=200
    )
    model.eval()

    # Save it to a temporary path to verify loading logic
    tmp_model_path = "tests/temp_test_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 1
    }, tmp_model_path)

    try:
        # 2. Setup Game State
        game = GameInstance()
        game.start_game()

        # 3. Simulate "Encoding" the state
        # In a real scenario, we would use a Tokenizer.
        # Here we mock the tokenization process as verified in minimal flow.
        # Just create a random token sequence representing the state.
        seq_len = 50
        fake_state_tokens = torch.randint(0, vocab_size, (1, seq_len))

        # 4. Load Model and Run Inference
        loaded_model = DuelTransformer(
            vocab_size=vocab_size,
            action_dim=action_dim,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            max_len=200
        )
        checkpoint = torch.load(tmp_model_path)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()

        with torch.no_grad():
            policy_logits, value = loaded_model(fake_state_tokens)

        # 5. Verify Outputs
        assert policy_logits.shape == (1, action_dim)
        assert value.shape == (1, 1)

        # 6. Select Action
        action_idx = torch.argmax(policy_logits, dim=1).item()
        assert 0 <= action_idx < action_dim

        print(f"Inference successful. Selected action index: {action_idx}, Value: {value.item()}")

    finally:
        # Cleanup
        if os.path.exists(tmp_model_path):
            os.remove(tmp_model_path)

def test_inference_with_batch():
    """Verify batch processing capability"""
    vocab_size = 1000
    action_dim = 600
    model = DuelTransformer(vocab_size=vocab_size, action_dim=action_dim, d_model=32, nhead=2, num_layers=1)
    model.eval()

    batch_size = 4
    seq_len = 30
    fake_batch = torch.randint(0, vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        policy, value = model(fake_batch)

    assert policy.shape == (batch_size, action_dim)
    assert value.shape == (batch_size, 1)
