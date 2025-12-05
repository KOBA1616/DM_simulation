
import sys
import os
sys.path.append(os.path.abspath("bin"))
try:
    import dm_ai_module
except ImportError:
    print("Failed to import dm_ai_module")
    sys.exit(1)

import pytest

# Constants
P1_ID = 0
P2_ID = 1

def setup_game():
    game_state = dm_ai_module.GameState(100) # 100 max cards
    return game_state

def test_draw_card():
    # Setup
    state = setup_game()
    card_db = {}

    # Add dummy cards to deck
    # Using exposed helper: add_card_to_deck(player_id, card_id, instance_id)
    state.add_card_to_deck(P1_ID, 1, 101)
    state.add_card_to_deck(P1_ID, 1, 102)

    initial_hand_count = len(state.players[P1_ID].hand)
    initial_deck_count = len(state.players[P1_ID].deck)

    assert initial_deck_count == 2

    # Direct invocation of GenericCardSystem.resolve_action for DRAW
    action_def = dm_ai_module.ActionDef()
    action_def.type = dm_ai_module.EffectActionType.DRAW_CARD
    action_def.value1 = 1

    dm_ai_module.GenericCardSystem.resolve_action(state, action_def, -1)

    assert len(state.players[P1_ID].hand) == initial_hand_count + 1
    assert len(state.players[P1_ID].deck) == initial_deck_count - 1
    assert state.players[P1_ID].hand[0].instance_id == 102 # Usually draws from top (back)

def test_mana_charge():
    state = setup_game()
    card_db = {}

    # Add card to hand
    state.add_card_to_hand(P1_ID, 1, 201)
    assert len(state.players[P1_ID].hand) == 1

    # Action: MANA_CHARGE
    action = dm_ai_module.Action()
    action.type = dm_ai_module.ActionType.MANA_CHARGE
    action.card_id = 1
    action.source_instance_id = 201 # Hand card to charge
    action.target_player = P1_ID

    # For MANA_CHARGE, EffectResolver handles it.
    dm_ai_module.EffectResolver.resolve_action(state, action, card_db)

    assert len(state.players[P1_ID].hand) == 0
    assert len(state.players[P1_ID].mana_zone) == 1
    assert state.players[P1_ID].mana_zone[0].instance_id == 201
    assert state.players[P1_ID].mana_zone[0].is_tapped == False

def test_tap_untap():
    state = setup_game()
    card_db = {}

    # Add creature to battle zone
    state.add_test_card_to_battle(P1_ID, 1, 301, False, False) # Untapped

    # Use GenericCardSystem for TAP action
    action_def = dm_ai_module.ActionDef()
    action_def.type = dm_ai_module.EffectActionType.TAP

    # Let's test UNTAP ALL SELF
    state.add_test_card_to_battle(P1_ID, 1, 302, True, False) # Tapped

    untap_action = dm_ai_module.ActionDef()
    untap_action.type = dm_ai_module.EffectActionType.UNTAP
    untap_action.target_choice = "ALL_SELF"

    dm_ai_module.GenericCardSystem.resolve_action(state, untap_action, -1)

    # Verify 302 is now untapped
    card_302 = None
    for c in state.players[P1_ID].battle_zone:
        if c.instance_id == 302:
            card_302 = c
            break
    assert card_302.is_tapped == False

    # Test TAP via Attack
    state.current_phase = dm_ai_module.Phase.ATTACK
    state.active_player_id = P1_ID

    attack_action = dm_ai_module.Action()
    attack_action.type = dm_ai_module.ActionType.ATTACK_PLAYER
    attack_action.source_instance_id = 301
    attack_action.target_player = P2_ID

    dm_ai_module.EffectResolver.resolve_action(state, attack_action, card_db)

    # Verify 301 is tapped
    card_301 = None
    for c in state.players[P1_ID].battle_zone:
        if c.instance_id == 301:
            card_301 = c
            break
    assert card_301.is_tapped == True

def test_break_shield():
    state = setup_game()
    card_db = {}

    # Add to deck first
    state.add_card_to_deck(P2_ID, 1, 401)
    # Move deck -> shield
    # DevTools expects Zone enum, not string!
    # Zone.DECK, Zone.SHIELD
    dm_ai_module.DevTools.move_cards(state, P2_ID, dm_ai_module.Zone.DECK, dm_ai_module.Zone.SHIELD, 1)

    assert len(state.players[P2_ID].shield_zone) == 1

    # Let's target self shield for atomic test using SEND_SHIELD_TO_GRAVE
    grave_def = dm_ai_module.ActionDef()
    grave_def.type = dm_ai_module.EffectActionType.SEND_SHIELD_TO_GRAVE

    state.active_player_id = P2_ID # Make P2 active so it affects P2 shields
    dm_ai_module.GenericCardSystem.resolve_action(state, grave_def, -1)

    assert len(state.players[P2_ID].shield_zone) == 0
    assert len(state.players[P2_ID].graveyard) == 1

def test_move_card_generic():
    state = setup_game()
    card_db = {}

    state.add_card_to_hand(P1_ID, 1, 501)

    # Use DevTools for atomic move verification
    # Zone.HAND, Zone.GRAVEYARD
    dm_ai_module.DevTools.move_cards(state, P1_ID, dm_ai_module.Zone.HAND, dm_ai_module.Zone.GRAVEYARD, 1)

    assert len(state.players[P1_ID].hand) == 0
    assert len(state.players[P1_ID].graveyard) == 1

if __name__ == "__main__":
    try:
        test_draw_card()
        print("test_draw_card PASS")
        test_mana_charge()
        print("test_mana_charge PASS")
        test_tap_untap()
        print("test_tap_untap PASS")
        test_break_shield()
        print("test_break_shield PASS")
        test_move_card_generic()
        print("test_move_card_generic PASS")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
