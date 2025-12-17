
import sys
import os

# Add the bin directory to sys.path to allow importing dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure the C++ module is built and in the bin directory.")
    sys.exit(1)

def verify_variable_linking():
    print("Verifying Variable Linking in CommandSystem (Direct Call)...")

    # 1. Initialize Game State
    state = dm_ai_module.GameState(100) # 100 cards

    # Register dummy cards for IDs used in test
    # ID 1, 2, 3 for Hand
    # ID 10-14 for Deck
    for i in range(1, 20):
        c = dm_ai_module.CardData(i, f"Card_{i}", 1, "FIRE", 1000, "CREATURE", [], [])
        dm_ai_module.register_card_data(c)

    # 2. Setup Cards
    player_id = 0
    # Player 0 Hand: 3 Cards (ID 1, 2, 3)
    dm_ai_module.GameState.add_card_to_hand(state, player_id, 1, 0)
    dm_ai_module.GameState.add_card_to_hand(state, player_id, 2, 1)
    dm_ai_module.GameState.add_card_to_hand(state, player_id, 3, 2)

    # Player 0 Deck: 5 Cards
    for i in range(5):
        dm_ai_module.GameState.add_card_to_deck(state, player_id, 10+i, 10+i)

    print(f"Initial Hand Size: {len(state.players[player_id].hand)}")
    print(f"Initial Deck Size: {len(state.players[player_id].deck)}")

    # 3. Create Commands

    # Command 1: Discard All from Hand -> output to "discard_count"
    cmd_discard = dm_ai_module.CommandDef()
    cmd_discard.type = dm_ai_module.JsonCommandType.DISCARD
    cmd_discard.target_group = dm_ai_module.TargetScope.PLAYER_SELF

    filter_hand = dm_ai_module.FilterDef()
    filter_hand.zones = ["HAND"]
    cmd_discard.target_filter = filter_hand

    cmd_discard.output_value_key = "discard_count"

    # Command 2: Draw Cards -> input from "discard_count"
    cmd_draw = dm_ai_module.CommandDef()
    cmd_draw.type = dm_ai_module.JsonCommandType.DRAW_CARD
    cmd_draw.input_value_key = "discard_count"

    # 4. Execute Commands Directy
    context = {}

    print("Executing DISCARD command...")
    context = dm_ai_module.CommandSystem.execute_command_with_context(state, cmd_discard, -1, player_id, context)

    print(f"Context after Discard: {context}")

    if "discard_count" not in context:
        print("FAILURE: 'discard_count' not found in execution context.")
        return

    discarded = context["discard_count"]
    print(f"Discarded count: {discarded}")

    current_hand = len(state.players[player_id].hand)
    print(f"Hand size after Discard: {current_hand}")

    if current_hand != 0:
        print(f"FAILURE: Expected hand size 0, got {current_hand}")

    if discarded != 3:
        print(f"FAILURE: Expected discard count 3, got {discarded}")

    print("Executing DRAW command...")
    context = dm_ai_module.CommandSystem.execute_command_with_context(state, cmd_draw, -1, player_id, context)

    final_hand = len(state.players[player_id].hand)
    print(f"Final Hand Size: {final_hand}")

    if final_hand != 3:
        print(f"FAILURE: Expected hand size 3, got {final_hand}")
    else:
        print("SUCCESS: Variable Linking Verified!")

if __name__ == "__main__":
    verify_variable_linking()
