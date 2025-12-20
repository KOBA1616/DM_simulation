
import sys
import os
import json

# Add the bin directory to sys.path to allow importing dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module
except ImportError:
    print("Error: Could not import dm_ai_module. Make sure the C++ module is built and in the bin directory.")
    sys.exit(1)

def verify_variable_linking():
    print("Verifying Variable Linking using ActionDef and PipelineExecutor...")

    # 1. Initialize Game State
    state = dm_ai_module.GameState(100) # 100 cards

    # Register dummy cards for IDs used in test
    for i in range(1, 20):
        # Create dummy cards.
        # Using def_static to register might not be directly available or needed if we inject DB.
        # But ActionHandler needs DB for filter checks.
        c = dm_ai_module.CardData(i, f"Card_{i}", 1, "FIRE", 1000, "CREATURE", [], [])
        dm_ai_module.register_card_data(c)

    # 2. Setup Cards
    player_id = state.active_player_id
    # Player Hand: 3 Cards (ID 1, 2, 3)
    dm_ai_module.GameState.add_card_to_hand(state, player_id, 1, 0)
    dm_ai_module.GameState.add_card_to_hand(state, player_id, 2, 1)
    dm_ai_module.GameState.add_card_to_hand(state, player_id, 3, 2)

    # Player Deck: 5 Cards
    for i in range(5):
        dm_ai_module.GameState.add_card_to_deck(state, player_id, 10+i, 10+i)

    print(f"Initial Hand Size: {len(state.players[player_id].hand)}")
    print(f"Initial Deck Size: {len(state.players[player_id].deck)}")

    # 3. Create Actions
    # We want to simulate:
    # 1. GET_STAT(HAND_COUNT) -> store in "hand_count_var"
    # 2. DISCARD ALL (Using "hand_count_var" as count? Or just DISCARD ALL logic)
    #    Actually, DiscardHandler supports "ALL" mode natively.
    #    But to test variable linking, let's try to pass a variable to DRAW.

    # Let's count cards in hand first.
    action_count = dm_ai_module.ActionDef()
    action_count.type = dm_ai_module.EffectActionType.GET_GAME_STAT
    action_count.str_val = "HAND_COUNT"
    action_count.output_value_key = "my_hand_count"

    # Then Discard All (just to clear hand so we can verify)
    action_discard = dm_ai_module.ActionDef()
    action_discard.type = dm_ai_module.EffectActionType.DISCARD
    action_discard.target_choice = "ALL"

    # Then Draw equal to "my_hand_count"
    action_draw = dm_ai_module.ActionDef()
    action_draw.type = dm_ai_module.EffectActionType.DRAW_CARD
    action_draw.input_value_key = "my_hand_count"

    # We need to execute these in a sequence.
    # We can use GenericCardSystem.resolve_action_with_context, but it takes ONE action.
    # To share context, we need to pass the updated context back.

    # We need to load CardDB because GenericCardSystem needs it
    # We don't have a direct get_all_cards binding that returns the map for Python to hold easily as map<int, Def>
    # But GameState doesn't hold it.
    # However, GenericCardSystem bindings might handle it internally if we pass dummy?
    # No, resolve_action_with_context takes db.

    # Workaround: Create a dummy map or rely on internal if binding supports it.
    # The binding `resolve_action_with_context` signature:
    # (state, source_id, action, db, ctx) -> ctx

    # We need to construct a DB map.
    # Since we cannot easily construct std::map<int, CardDefinition> in Python without a helper,
    # let's assume we can pass an empty dict if the handlers don't strictly need it for basic actions.
    # But DiscardHandler checks DB for validity.

    # Let's try to get the DB from JSON loader?
    # JsonLoader.load_cards returns void, loads into Registry.
    # The binding for resolve_action_with_context expects `std::map`.
    # Pybind11 converts python dict to std::map.
    # So we can build a python dict.

    card_db = {}
    for i in range(1, 20):
        # We need to match CardDefinition struct.
        # Python binding for CardDefinition constructor:
        # (id, name, civ, races, cost, power, keywords, effects)
        kw = dm_ai_module.CardKeywords()
        # id, name, civ, races, cost, power, kw, effects
        c_def = dm_ai_module.CardDefinition(i, f"Card_{i}", "FIRE", [], 1, 1000, kw, [])
        card_db[i] = c_def

    context = {}

    print("\n--- Executing GET_GAME_STAT (HAND_COUNT) ---")
    context = dm_ai_module.GenericCardSystem.resolve_action_with_context(state, -1, action_count, card_db, context)
    print(f"Context: {context}")

    if "my_hand_count" not in context:
        print("FAILURE: 'my_hand_count' not found in context.")
        return

    count_val = context["my_hand_count"]
    print(f"Hand Count Recorded: {count_val}")
    if count_val != 3:
        print(f"FAILURE: Expected 3, got {count_val}")

    print("\n--- Executing DISCARD ALL ---")
    context = dm_ai_module.GenericCardSystem.resolve_action_with_context(state, -1, action_discard, card_db, context)

    current_hand = len(state.players[player_id].hand)
    print(f"Hand Size after Discard: {current_hand}")
    if current_hand != 0:
        print("FAILURE: Hand should be empty.")

    print("\n--- Executing DRAW (Using my_hand_count) ---")
    context = dm_ai_module.GenericCardSystem.resolve_action_with_context(state, -1, action_draw, card_db, context)

    final_hand = len(state.players[player_id].hand)
    print(f"Final Hand Size: {final_hand}")

    if final_hand != 3:
        print(f"FAILURE: Expected 3, got {final_hand}")
    else:
        print("SUCCESS: Variable Linking Verified!")

if __name__ == "__main__":
    verify_variable_linking()
