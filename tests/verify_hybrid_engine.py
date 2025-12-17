import sys
import os
import json
import traceback

# Add build directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

try:
    import dm_ai_module
except ImportError as e:
    print(f"Error: Could not import dm_ai_module. {e}")
    sys.exit(1)

def test_hybrid_engine():
    print("Initializing GameState...")
    state = dm_ai_module.GameState(100)
    state.setup_test_duel()

    # 1. Test New Command Integration (Hybrid)
    print("\n--- Test 1: New Command Execution (DRAW_CARD) ---")

    # Define a command manually (as per Requirement 00: no auto conversion)
    cmd = dm_ai_module.CommandDef()
    cmd.type = dm_ai_module.JsonCommandType.DRAW_CARD
    cmd.amount = 2

    effect = dm_ai_module.EffectDef(
        dm_ai_module.TriggerType.ON_PLAY,
        dm_ai_module.ConditionDef(),
        []  # Empty actions (Legacy)
    )

    try:
        # Note: Pybind11 exposes std::vector as a copy (list). Appending to the getter result does nothing.
        # We must assign the list back to the property.
        effect.commands = [cmd]
    except AttributeError:
        print("FAIL: EffectDef does not expose 'commands' attribute in Python.")
        return False

    card_data = dm_ai_module.CardData(
        1001, "Hybrid Draw Card", 5, "WATER", 3000, "CREATURE", [], [effect]
    )

    dm_ai_module.register_card_data(card_data)

    # Verify Data Integrity
    card_reg = dm_ai_module.CardRegistry.get_all_definitions()
    card_def = card_reg[1001]

    if len(card_def.effects[0].commands) != 1:
        print(f"FAIL: Commands not registered. Count: {len(card_def.effects[0].commands)}")
        return False

    print("Card registered with Command successfully.")

    # Execution Test
    # Manually populate deck
    for i in range(5):
        state.add_card_to_deck(0, 1, 100+i)

    # We need a dummy instance ID.
    state.add_card_to_hand(0, 1001, 999) # Add card to hand to have an instance

    initial_hand = len(state.players[0].hand) # Should be 1

    print("Executing resolve_effect...")
    # Note: We need to pass the card_db to resolve_effect
    card_db = dm_ai_module.CardRegistry.get_all_definitions()
    # Use resolve_effect_with_db if available, or resolve_effect if it defaults
    # Binding shows: resolve_effect_with_db for explicit db
    dm_ai_module.GenericCardSystem.resolve_effect_with_db(state, card_def.effects[0], 999, card_db)

    new_hand = len(state.players[0].hand)
    print(f"Hand: {initial_hand} -> {new_hand}")

    if new_hand != initial_hand + 2:
        print(f"FAIL: Hand did not increase by 2. Diff: {new_hand - initial_hand}")
        return False

    print("PASS: Hybrid Engine executed Command via GenericCardSystem.")

    return True

if __name__ == "__main__":
    if test_hybrid_engine():
        print("\nAll Tests Passed.")
        sys.exit(0)
    else:
        print("\nTests Failed.")
        sys.exit(1)
