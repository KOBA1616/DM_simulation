import os
import sys
import torch
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import module
try:
    import dm_ai_module
except ImportError:
    print(f"Failed to import dm_ai_module from {bin_path}")
    sys.exit(1)

# Import Network
try:
    from dm_toolkit.ai.agent.network import AlphaZeroTransformer
except ImportError as e:
    print(f"Failed to import AlphaZeroTransformer: {e}")
    sys.exit(1)

def verify_transformer_integration():
    print("--- Verifying Phase 4 Transformer Integration ---")

    # 1. Initialize Game State
    print("Initializing GameState...")
    cards_path = os.path.join(project_root, "data/cards.json")
    if not os.path.exists(cards_path):
        print(f"Error: cards.json not found at {cards_path}")
        sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    # Create game
    game_state = dm_ai_module.GameState(1000)
    dm_ai_module.PhaseManager.start_game(game_state, card_db)

    # 2. Convert to Sequence (TensorConverter V2)
    print("Testing TensorConverter.convert_to_sequence...")
    try:
        tokens = dm_ai_module.TensorConverter.convert_to_sequence(game_state, 0, card_db, True)
        print(f"Tokens generated: {len(tokens)}")
        print(f"Sample tokens: {tokens[:10]}...")

        # Verify Token Constants
        assert len(tokens) <= dm_ai_module.TensorConverter.MAX_SEQ_LEN, "Sequence length exceeds MAX_SEQ_LEN"

    except Exception as e:
        print(f"Error in convert_to_sequence: {e}")
        sys.exit(1)

    # 3. Instantiate Transformer Model
    print("Instantiating AlphaZeroTransformer...")
    vocab_size = dm_ai_module.TensorConverter.VOCAB_SIZE
    max_seq_len = dm_ai_module.TensorConverter.MAX_SEQ_LEN
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

    print(f"Model Config: Vocab={vocab_size}, MaxSeq={max_seq_len}, ActionSpace={action_size}")

    model = AlphaZeroTransformer(
        action_size=action_size,
        embedding_dim=64, # Small for test
        depth=2,
        heads=4,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len
    )
    model.eval()

    # 4. Forward Pass
    print("Running Forward Pass...")
    try:
        # Prepare input tensor [Batch, SeqLen]
        input_tensor = torch.tensor([tokens], dtype=torch.long)

        # Valid mask (all valid for this test)
        mask = torch.ones((1, len(tokens)), dtype=torch.bool)

        policy_logits, value = model(input_tensor, mask)

        print(f"Policy Shape: {policy_logits.shape}")
        print(f"Value Shape: {value.shape}")

        assert policy_logits.shape == (1, action_size), f"Expected policy shape (1, {action_size}), got {policy_logits.shape}"
        assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"

        print("Forward pass successful.")

    except Exception as e:
        print(f"Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("--- Verification Complete: SUCCESS ---")

if __name__ == "__main__":
    verify_transformer_integration()
