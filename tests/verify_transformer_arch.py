
import sys
import os
import torch
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

try:
    import dm_ai_module
    print("dm_ai_module loaded.")
except ImportError:
    print("Failed to load dm_ai_module")
    sys.exit(1)

from dm_toolkit.ai.agent.transformer_model import DuelTransformer

def verify_transformer_pipeline():
    print("\n--- Verifying Transformer Architecture ---")

    # 1. Setup Game State
    state = dm_ai_module.GameState(100)
    state.add_card_to_hand(0, 1, 0)
    state.add_card_to_mana(0, 2, 1)
    # Using add_test_card_to_battle as revealed by grep
    state.add_test_card_to_battle(1, 3, 2, False, False) # Opponent creature

    # 2. Tokenize
    vocab_size = dm_ai_module.TokenConverter.get_vocab_size()
    tokens = dm_ai_module.TokenConverter.encode_state(state, 0, 128)
    print(f"Generated {len(tokens)} tokens.")

    # Create batch
    input_tensor = torch.tensor([tokens], dtype=torch.long) # (1, Seq)

    # 3. Model Inference
    action_dim = 600
    model = DuelTransformer(vocab_size=vocab_size, action_dim=action_dim, d_model=64, num_layers=2)
    model.eval()

    print("Running forward pass...")
    with torch.no_grad():
        policy, value = model(input_tensor)

    print(f"Policy Shape: {policy.shape} (Expected: [1, {action_dim}])")
    print(f"Value Shape: {value.shape} (Expected: [1, 1])")

    assert policy.shape == (1, action_dim)
    assert value.shape == (1, 1)

    # 4. Synergy Matrix Check
    print("Checking Synergy Matrix integration...")
    bias = model.synergy_graph.get_bias_for_sequence(input_tensor)
    print(f"Synergy Bias Shape: {bias.shape} (Expected: [1, {len(tokens)}, {len(tokens)}])")
    assert bias.shape == (1, len(tokens), len(tokens))

    print("\n--- Verification Successful ---")

if __name__ == "__main__":
    verify_transformer_pipeline()
