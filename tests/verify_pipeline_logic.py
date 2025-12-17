
import sys
import os

# Add bin path to load dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))

try:
    import dm_ai_module
    from dm_ai_module import GameState, CardData, Civilization, CardType, EffectSystem, ConditionDef, EffectDef, ActionDef, EffectActionType, TargetScope, TriggerType
except ImportError as e:
    print(f"Failed to import dm_ai_module: {e}")
    print("Ensure dm_ai_module is built and in the python path (e.g. bin/).")
    sys.exit(1)

def verify_pipeline_logic():
    print("Verifying Pipeline Logic (Phase 6 Step 3)...")

    # Initialize GameState
    game = GameState(40) # 40 cards total

    # Define a test card with Pipeline enabled via stat_key hack
    # Effect: If Mana >= 3 (MANA_ARMED 3), Draw 1 card.

    card_id = 9001
    card_name = "PipelineTester"

    # Condition: MANA_ARMED 3
    # Note: In Python bindings, we might need to verify how ConditionDef is constructed.
    # Assuming standard dict or object based on bindings.

    # Construct Effect
    eff = EffectDef()
    eff.trigger = TriggerType.ON_PLAY

    cond = ConditionDef()
    cond.type = "MANA_ARMED"
    cond.value = 3
    cond.str_val = "FIRE" # Requires Fire Mana
    cond.stat_key = "ENABLE_PIPELINE" # Trigger the pipeline path
    eff.condition = cond

    act = ActionDef()
    act.type = EffectActionType.DRAW_CARD
    act.value1 = 1 # Draw 1
    act.scope = TargetScope.NONE
    eff.actions = [act]

    # Register Card
    # We need to create a CardData object.
    # Constructor: CardData(id, name, cost, civ_str, power, type_str, races, effects)
    # Note: CardType should be passed as string "CREATURE" based on binding error.
    card_data = CardData(card_id, card_name, 3, "FIRE", 3000, "CREATURE", ["FireBird"], [eff])

    dm_ai_module.register_card_data(card_data)

    # Scenario 1: Condition Not Met (Mana < 3)
    print("\n--- Scenario 1: Condition Not Met (Mana < 3) ---")
    game = GameState(40)
    p0 = game.players[0]

    # Setup: 1 Fire Mana
    # game.add_card_to_mana(0, card_id, 100) # Need card definitions for civ check in logic.
    # The registered card is FIRE.

    # Manually adding cards using helper if exposed
    # add_card_to_mana(pid, cid, iid)
    game.add_card_to_mana(0, card_id, 100)

    # Hand: The test card
    game.add_card_to_hand(0, card_id, 200)

    prev_hand_size = len(p0.hand)
    print(f"Hand size before: {prev_hand_size}")

    # Using GenericCardSystem.resolve_trigger(game, ON_PLAY, instance_id, db)
    # card_db = dm_ai_module.get_all_cards() # Not exposed directly

    # Construct local card_db for the test
    # We need the CardDefinition, not just CardData.
    # But GenericCardSystem expects a map of ID -> CardDefinition.
    # In Python bindings, CardData IS the CardDefinition wrapper usually.
    # Let's verify if we can fetch it.

    # Assuming dm_ai_module.get_card_data exists (or CardRegistry.get_card_data)
    # If not, we use the object we created: card_data.
    # But get_card_data returns CardDefinition (C++) wrapped. card_data is CardData (C++ wrapper).
    # They are likely the same type in bindings.

    card_db = {card_id: card_data}

    # We need to make sure the card is in the battle zone for ON_PLAY trigger context
    game.add_test_card_to_battle(0, card_id, 200, False, True)

    # Resolve Effect Directly to test pipeline
    print("Invoking resolve_effect directly...")
    # Binding signature suggests no card_db needed (auto-fetched or not required for simple effects?)
    EffectSystem.resolve_effect(game, eff, 200)

    curr_hand_size = len(p0.hand)
    print(f"Hand size after: {curr_hand_size}")

    if curr_hand_size == prev_hand_size:
        print("SUCCESS: Card not drawn (Condition correctly failed).")
    else:
        print("FAILURE: Card drawn despite condition failure.")

    # Scenario 2: Condition Met (Mana >= 3)
    print("\n--- Scenario 2: Condition Met (Mana >= 3) ---")
    game = GameState(40)
    p0 = game.players[0]

    # Setup: 3 Fire Mana
    game.add_card_to_mana(0, card_id, 101)
    game.add_card_to_mana(0, card_id, 102)
    game.add_card_to_mana(0, card_id, 103)

    game.add_test_card_to_battle(0, card_id, 201, False, True)

    # Add cards to deck for drawing
    game.add_card_to_deck(0, card_id, 301)
    game.add_card_to_deck(0, card_id, 302)

    prev_hand_size = len(p0.hand)
    print(f"Hand size before: {prev_hand_size}")

    print("Invoking resolve_effect directly...")
    EffectSystem.resolve_effect(game, eff, 201)

    curr_hand_size = len(p0.hand)
    print(f"Hand size after: {curr_hand_size}")

    if curr_hand_size == prev_hand_size + 1:
        print("SUCCESS: Card drawn (Condition correctly met via Pipeline).")
    else:
        print("FAILURE: Card not drawn.")

if __name__ == "__main__":
    verify_pipeline_logic()
