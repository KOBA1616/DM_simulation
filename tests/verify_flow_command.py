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

def test_flow_logic():
    print("Initializing GameState...")
    state = dm_ai_module.GameState(100)
    state.setup_test_duel()

    # Load real card definitions to ensure Bronze-Arm Tribe (ID 1) exists and has NATURE
    card_reg = dm_ai_module.CardRegistry.get_all_definitions()
    if 1 not in card_reg:
        print("Loading cards.json...")
        dm_ai_module.JsonLoader.load_cards("data/cards.json")
        card_reg = dm_ai_module.CardRegistry.get_all_definitions()

    # Define Conditional Command (Condition is NOT met)
    # If Mana Armed 3 (NATURE) -> Draw 2, Else -> Draw 1

    # Branch 1: Draw 2
    cmd_true = dm_ai_module.CommandDef()
    cmd_true.type = dm_ai_module.JsonCommandType.DRAW_CARD
    cmd_true.amount = 2

    # Branch 2: Draw 1
    cmd_false = dm_ai_module.CommandDef()
    cmd_false.type = dm_ai_module.JsonCommandType.DRAW_CARD
    cmd_false.amount = 1

    # Main Command: FLOW
    cmd_flow = dm_ai_module.CommandDef()
    cmd_flow.type = dm_ai_module.JsonCommandType.FLOW

    cond = dm_ai_module.ConditionDef()
    cond.type = "MANA_ARMED"
    cond.value = 3
    cond.str_val = "NATURE"

    cmd_flow.condition = cond
    cmd_flow.if_true = [cmd_true]
    cmd_flow.if_false = [cmd_false]

    effect = dm_ai_module.EffectDef(
        dm_ai_module.TriggerType.ON_PLAY,
        dm_ai_module.ConditionDef(), # Effect-level condition
        []
    )
    effect.commands = [cmd_flow]

    card_data = dm_ai_module.CardData(
        2002, "Flow Tester", 5, "NATURE", 3000, "CREATURE", [], [effect]
    )
    dm_ai_module.register_card_data(card_data)

    # Refresh reg
    card_reg = dm_ai_module.CardRegistry.get_all_definitions()
    card_def = card_reg[2002]

    # Test 1: Condition False (No Mana)
    print("\n--- Test 1: Condition False (No Mana) ---")
    state.add_card_to_hand(0, 2002, 999)
    state.add_card_to_deck(0, 1, 100)
    state.add_card_to_deck(0, 1, 101)

    initial_hand = len(state.players[0].hand)

    dm_ai_module.GenericCardSystem.resolve_effect_with_db(state, card_def.effects[0], 999, card_reg)

    new_hand = len(state.players[0].hand)
    print(f"Hand: {initial_hand} -> {new_hand}")

    # Expect +1 (False branch)
    if new_hand != initial_hand + 1:
        print(f"FAIL: Hand increased by {new_hand - initial_hand}, expected 1")
        return False

    print("PASS: Flow False Branch executed.")

    # Test 2: Condition True (Add Mana)
    print("\n--- Test 2: Condition True (Mana Armed 3) ---")
    state.add_card_to_mana(0, 1, 200) # Bronze-Arm Tribe is Nature
    state.add_card_to_mana(0, 1, 201)
    state.add_card_to_mana(0, 1, 202)

    # Verify Mana
    mana_count = len(state.players[0].mana_zone)
    print(f"Mana Count: {mana_count}")

    initial_hand = new_hand
    # Add more cards to deck
    state.add_card_to_deck(0, 1, 102)
    state.add_card_to_deck(0, 1, 103)

    dm_ai_module.GenericCardSystem.resolve_effect_with_db(state, card_def.effects[0], 999, card_reg)

    new_hand = len(state.players[0].hand)
    print(f"Hand: {initial_hand} -> {new_hand}")

    # Expect +2 (True branch)
    if new_hand != initial_hand + 2:
        print(f"FAIL: Hand increased by {new_hand - initial_hand}, expected 2")
        return False

    print("PASS: Flow True Branch executed.")
    return True

if __name__ == "__main__":
    if test_flow_logic():
        print("\nAll Tests Passed.")
        sys.exit(0)
    else:
        print("\nTests Failed.")
        sys.exit(1)
