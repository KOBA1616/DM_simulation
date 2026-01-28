
import sys
import os
import torch
import traceback

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.ai.agent.mcts import MCTS
from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit import dm_ai_module
from dm_toolkit.dm_ai_module import GameState

def run():
    print("--- Reproduction Script: MCTS + Transformer (Default Config) ---")

    # 1. Setup Dummy Dependencies
    try:
        # Try loading real cards if available, else dummy
        card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")
    except Exception:
        print("Warning: Could not load cards.json, using empty DB.")
        card_db = {}

    # 2. Instantiate Model
    print("Initializing DuelTransformer...")
    vocab_size = 2000
    action_dim = 600
    model = DuelTransformer(vocab_size=vocab_size, action_dim=action_dim)
    model.eval()

    # 3. Instantiate MCTS (WITHOUT state_converter)
    # This should trigger the default TensorConverter which outputs floats
    print("Initializing MCTS (Default Configuration)...")
    mcts = MCTS(network=model, card_db=card_db, simulations=2)

    # 4. Create State
    state = GameState()
    # Initialize some dummy deck to avoid empty deck issues if engine checks
    deck = [1] * 40
    state.set_deck(0, deck)
    state.set_deck(1, deck)

    # 5. Run Search
    print("Attempting mcts.search(state)...")
    try:
        root = mcts.search(state)
        print("CRITICAL: Search completed successfully! Issue NOT reproduced.")
    except Exception as e:
        print("\nSUCCESS: Caught expected exception!")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run()
