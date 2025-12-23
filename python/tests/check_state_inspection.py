
import sys
import os

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError as e:
    print(f"Error importing dm_ai_module: {e}")
    sys.exit(1)

def inspect_state():
    state = dm_ai_module.GameState(100)
    # Add some dummy cards
    # state.add_card_to_hand(player_id, card_id, instance_id)
    # Using small IDs to be safe
    state.add_card_to_hand(0, 1, 0)
    state.add_card_to_hand(0, 2, 1)
    state.add_card_to_mana(0, 3, 2)
    state.add_test_card_to_battle(0, 4, 3, False, False)

    # Check if we can access players
    if hasattr(state, 'players'):
        print("state.players exists")
        p0 = state.players[0]
        print(f"Player 0 hand size: {len(p0.hand)}")
        if len(p0.hand) > 0:
            c = p0.hand[0]
            print(f"Card in hand: {c}")
            # Check if it has card_id attribute
            if hasattr(c, 'card_id'):
                print(f"Card ID: {c.card_id}")
    else:
        print("state.players does NOT exist")

    # Check other accessors
    if hasattr(state, 'get_card_instance'):
        print("state.get_card_instance exists")

    if hasattr(state, 'get_player'):
        print("state.get_player exists")

if __name__ == "__main__":
    inspect_state()
