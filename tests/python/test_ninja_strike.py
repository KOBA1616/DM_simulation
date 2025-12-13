
import sys
import os
sys.path.append(os.path.abspath("bin"))
try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module")
    sys.exit(1)

import pytest
import json

def test_ninja_strike_flow():
    # 1. Load Cards
    loader = dm_ai_module.JsonLoader

    # Ensure data/ninja_test.json exists
    ninja_json_path = "data/ninja_test.json"
    ninja_data = [
        {
            "id": 9999,
            "name": "Ninja Strike Unit",
            "type": "CREATURE",
            "civilization": "LIGHT",
            "races": ["Shinobi"],
            "cost": 7,
            "power": 5000,
            "keywords": {},
            "effects": [],
            "reaction_abilities": [
                {
                    "type": "NINJA_STRIKE",
                    "cost": 7,
                    "zone": "HAND",
                    "condition": {
                        "trigger_event": "ON_BLOCK_OR_ATTACK",
                        "civilization_match": True,
                        "mana_count_min": 7
                    }
                }
            ]
        }
    ]
    with open(ninja_json_path, 'w') as f:
        json.dump(ninja_data, f)

    cards = loader.load_cards(ninja_json_path)
    if 9999 not in cards:
        pytest.fail("NinjaTest card not found")

    # 2. Setup Game State
    game_state = dm_ai_module.GameState(42)
    card_db = cards

    # Register data for registry
    for cid, cdef in card_db.items():
        # JsonLoader creates CardData with correct types, but when creating explicitly we need string for civ.
        # But JsonLoader already loaded it?
        # JsonLoader loads into a map.
        # We need to register for Generic functions.
        # Use helper:
        cdata = dm_ai_module.CardData(cid, cdef.name, cdef.cost,
                                          cdef.civilizations, # Use list of Enums
                                      cdef.power, "CREATURE", cdef.races, [],
                                      cdef.reaction_abilities) # Pass reaction abilities!
        dm_ai_module.register_card_data(cdata)

    # Initialize players
    p1 = game_state.players[0]
    p2 = game_state.players[1]

    # Give P1 an attacker
    attacker_card_id = 9999

    # We need to manually add it to battle zone since we don't have a distinct attacker card in json
    # Use helper
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

    # 7. Generate Actions (Should see DECLARE_REACTION)
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game_state, card_db)
    ninja_action = None
    for a in actions:
        print(f"Action: {a.type}, Source: {a.source_instance_id}")
        if a.type == dm_ai_module.ActionType.DECLARE_REACTION:
            ninja_action = a

    # If Ninja Strike fails, check civ
        print(f"Card Civ: {cards[9999].civilizations[0]}")

    assert ninja_action is not None, "Should have Ninja Strike action"

    # 8. Use Ninja Strike
    dm_ai_module.EffectResolver.resolve_action(game_state, ninja_action, card_db)

    # 9. Verify Ninja in Battle Zone
    p2_battle = game_state.players[1].battle_zone
    has_ninja = False
    for c in p2_battle:
        if c.instance_id == 300:
            has_ninja = True
            break

    assert has_ninja, "Ninja should be summoned"

    print("Test Passed: Ninja Strike executed successfully")

if __name__ == "__main__":
    test_ninja_strike_flow()
