
import sys
import os
sys.path.append(os.path.abspath("bin"))
try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module")
    sys.exit(1)

import pytest

def test_ninja_strike_flow():
    # 1. Load Cards
    loader = dm_ai_module.JsonLoader
    # Ensure data/ninja_test.json exists (created in earlier step)
    cards = loader.load_cards("data/ninja_test.json")
    if 9999 not in cards:
        pytest.fail("NinjaTest card not found")

    # 2. Setup Game State
    game_state = dm_ai_module.GameState(42)
    card_db = cards

    # Initialize players
    p1 = game_state.players[0]
    p2 = game_state.players[1]

    # Give P1 an attacker
    # Assuming ID 1 exists as dummy or use 9999 modified
    # We need a valid attacker definition. Let's create a dummy attacker if needed or use 9999.
    attacker_card_id = 9999

    # Setup P1 Battle Zone
    # Use helper if available, or manual setup via python bindings if exposed
    # Since direct member access to vector might be copy-only, use exposed methods if any.
    # bindings.cpp says we can access battle_zone.
    # But modifying the list from python usually doesn't reflect in C++ unless using wrappers.
    # We need "add_card_to_battle" or similar.
    # Or we can use "move_card" from DevTools if exposed?
    # dm_ai_module.DevTools.move_cards?

    # Let's try to set up using add_card_to_battle helper if it exists in bindings.
    # If not, we might struggle to set up state.
    # Looking at memory, `add_test_card_to_battle` exists.

    game_state.add_test_card_to_battle(0, attacker_card_id, 100, False, False) # P1, ID, InstanceID, Tapped, Sick

    # Setup P2 Hand with Ninja Strike
    # Ninja Test: Cost 7, Civil LIGHT.
    # Condition: Mana >= 7, Civil Match.
    # Setup P2 Mana
    for i in range(7):
        game_state.add_card_to_mana(1, attacker_card_id, 200+i) # P2, ID, InstanceID

    game_state.add_card_to_hand(1, attacker_card_id, 300) # The Ninja

    # 3. Start Turn / Phase
    game_state.active_player_id = 0
    game_state.current_phase = dm_ai_module.Phase.ATTACK

    # 4. Generate Attack Action
    # P1 attacks P2
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game_state, card_db)
    attack_action = None
    for a in actions:
        if a.type == dm_ai_module.ActionType.ATTACK_PLAYER:
            attack_action = a
            break

    assert attack_action is not None, "Should be able to attack"

    # 5. Resolve Attack -> Should Trigger Reaction Window Check
    dm_ai_module.EffectResolver.resolve_action(game_state, attack_action, card_db)

    # 6. Check Pending Effects
    # We expect a REACTION_WINDOW pending effect
    pending = dm_ai_module.get_pending_effects_info(game_state)
    print("Pending Effects:", pending)

    # Check if we have REACTION_WINDOW (Enum value might be exposed as int or property)
    # EffectType.REACTION_WINDOW was added at the end.
    # Need to verify if python bindings updated automatically?
    # Bindings usually need update if they map enums explicitly.
    # I didn't update `src/python/bindings.cpp`. This is a risk.
    # If I didn't update bindings, I can't check Enum value by name.
    # But the C++ code runs.

    # 7. Generate Actions (Should see DECLARE_REACTION)
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game_state, card_db)
    ninja_action = None
    for a in actions:
        print(f"Action: {a.type}, Source: {a.source_instance_id}")
        # If bindings not updated, ActionType.DECLARE_REACTION might not exist in Python.
        # It will show as generic int or error if strict enum.
        if a.type == dm_ai_module.ActionType.DECLARE_REACTION:
            ninja_action = a

    assert ninja_action is not None, "Should have Ninja Strike action"

    # 8. Use Ninja Strike
    dm_ai_module.EffectResolver.resolve_action(game_state, ninja_action, card_db)

    # 9. Verify Ninja in Battle Zone
    p2_battle = game_state.players[1].battle_zone
    # Since bindings return copy, we fetch again.
    has_ninja = False
    for c in p2_battle:
        if c.instance_id == 300:
            has_ninja = True
            break

    assert has_ninja, "Ninja should be summoned"

    print("Test Passed: Ninja Strike executed successfully")

if __name__ == "__main__":
    test_ninja_strike_flow()
