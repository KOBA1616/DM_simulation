import dm_ai_module
import torch
import os
import sys

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_ai.agent.network import AlphaZeroNetwork
from py_ai.agent.mcts import MCTS

# Load DB
db_path = os.path.join(os.path.dirname(__file__), '..', "data", "cards.csv")
if not os.path.exists(db_path):
    print("Error: cards.csv not found")
    sys.exit(1)

try:
    card_db = dm_ai_module.CsvLoader.load_cards(db_path)
    print(f"Loaded {len(card_db)} cards")

    # Init Game
    gs = dm_ai_module.GameState(123)
    gs.setup_test_duel()
    dm_ai_module.PhaseManager.start_game(gs)

    # Init Network
    input_size = dm_ai_module.TensorConverter.INPUT_SIZE
    action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE
    net = AlphaZeroNetwork(input_size, action_size)

    # Init MCTS
    mcts = MCTS(net, card_db, simulations=10)

    # Run Search
    print("Running MCTS...")
    root = mcts.search(gs)
    print(f"Root visits: {root.visit_count}")
    print(f"Children: {len(root.children)}")
    if root.children:
        best_child = mcts._select_child(root)
        if best_child:
            print(f"Best Action: {best_child.action.to_string()}")
        else:
            print("No best child found")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
