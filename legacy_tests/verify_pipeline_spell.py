import sys
import os

# Add the bin directory to sys.path
bin_path = os.path.join(os.getcwd(), 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module. Make sure the module is built and in the bin directory.")
    sys.exit(1)

def verify_pipeline_spell():
    print("Verifying Pipeline CAST_SPELL Resolution...")

    # 1. Setup GameState
    state = dm_ai_module.GameState(100)
    state.active_player_id = 0

    # 2. Setup Cards
    # Spell (P0): ID=200, Inst=10, Cost 2
    state.add_card_to_hand(0, 200, 10)

    # 3. Define Card Data
    effect_def = dm_ai_module.EffectDef()
    action_def = dm_ai_module.ActionDef()
    action_def.type = dm_ai_module.EffectActionType.ADD_MANA
    # Add 1 mana to self
    # We need to set target correctly. ADD_MANA takes targets.
    # Usually "Put top card of deck into mana".
    # This might require complex setup.
    # Let's try simpler: TAP_TARGET (Creature).
    # But we need a target.

    # Let's use DRAW_CARD. Simpler.
    action_def.type = dm_ai_module.EffectActionType.DRAW_CARD
    action_def.value1 = 1

    effect_def.actions.append(action_def)

    spell_def = dm_ai_module.CardDefinition(200, "Test Spell", "WATER", [], 2, 0, dm_ai_module.CardKeywords(), [effect_def])
    spell_def.type = dm_ai_module.CardType.SPELL

    card_db = {
        200: spell_def
    }

    # Add dummy deck to draw from
    state.add_card_to_deck(0, 999, 50)
    state.add_card_to_deck(0, 999, 51)

    # 4. Invoke PLAY_CARD action via Pipeline
    action = dm_ai_module.Action()
    action.type = dm_ai_module.ActionType.PLAY_CARD
    action.source_instance_id = 10

    print("Executing PLAY_CARD Action...")
    dm_ai_module.EffectResolver.resolve_action(state, action, card_db)

    # 5. Check results
    # Expectation: Spell (Inst 10) in Graveyard. Hand size increased (Draw).

    p0_grave = state.get_zone(0, dm_ai_module.Zone.GRAVEYARD)
    p0_hand = state.get_zone(0, dm_ai_module.Zone.HAND)
    p0_stack = state.get_zone(0, dm_ai_module.Zone.STACK)

    found_in_grave = 10 in p0_grave
    found_in_hand = 10 in p0_hand
    found_in_stack = 10 in p0_stack

    print(f"Spell in Graveyard: {found_in_grave}")
    print(f"Spell in Hand: {found_in_hand}")
    print(f"Spell in Stack: {found_in_stack}")
    print(f"Cards in Hand: {len(p0_hand)}") # Should be start (1) - spell (1) + draw (1) = 1 (but new card)

    # Check if drawn card is in hand (Inst 50 or 51)
    drawn_50 = 50 in p0_hand
    drawn_51 = 51 in p0_hand

    print(f"Drawn 50: {drawn_50}")
    print(f"Drawn 51: {drawn_51}")

    if found_in_grave and (drawn_50 or drawn_51):
        print("SUCCESS: Spell resolved correctly.")
    else:
        print("FAILURE: Spell resolution failed.")
        if not found_in_grave:
             print("Reason: Spell not in graveyard.")
        if not (drawn_50 or drawn_51):
             print("Reason: Effect (Draw) did not happen.")

if __name__ == "__main__":
    verify_pipeline_spell()
