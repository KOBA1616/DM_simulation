
import sys
import os
sys.path.append(os.path.abspath("bin"))
try:
    import dm_ai_module
    from dm_ai_module import GameState, CardData, EffectDef, ActionDef, ConditionDef, TriggerType, EffectActionType, TargetScope, CardDefinition
except ImportError:
    print("Failed to import dm_ai_module")
    sys.exit(1)

import pytest

# Constants
P1_ID = 0
P2_ID = 1

def setup_game():
    game_state = dm_ai_module.GameState(100) # 100 max cards
    # game_state.setup_test_players() # Removed as not exposed
    return game_state

def test_draw_card():
    state = setup_game()
    card_db = {}
    state.add_card_to_deck(P1_ID, 1, 101)
    state.add_card_to_deck(P1_ID, 1, 102)
    initial_hand_count = len(state.players[P1_ID].hand)
    initial_deck_count = len(state.players[P1_ID].deck)
    action_def = dm_ai_module.ActionDef()
    action_def.type = dm_ai_module.EffectActionType.DRAW_CARD
    action_def.value1 = 1
    dm_ai_module.GenericCardSystem.resolve_action(state, action_def, -1)
    assert len(state.players[P1_ID].hand) == initial_hand_count + 1
    assert len(state.players[P1_ID].deck) == initial_deck_count - 1
    assert state.players[P1_ID].hand[0].instance_id == 102

def test_mana_charge():
    state = setup_game()
    card_db = {}
    state.add_card_to_hand(P1_ID, 1, 201)
    action = dm_ai_module.Action()
    action.type = dm_ai_module.ActionType.MANA_CHARGE
    action.card_id = 1
    action.source_instance_id = 201
    action.target_player = P1_ID
    dm_ai_module.EffectResolver.resolve_action(state, action, card_db)
    assert len(state.players[P1_ID].hand) == 0
    assert len(state.players[P1_ID].mana_zone) == 1
    assert state.players[P1_ID].mana_zone[0].instance_id == 201

def test_tap_untap():
    state = setup_game()
    card_db = {}
    state.add_test_card_to_battle(P1_ID, 1, 301, False, False)
    state.add_test_card_to_battle(P1_ID, 1, 302, True, False)
    untap_action = dm_ai_module.ActionDef()
    untap_action.type = dm_ai_module.EffectActionType.UNTAP
    untap_action.target_choice = "ALL_SELF"
    dm_ai_module.GenericCardSystem.resolve_action(state, untap_action, -1)
    card_302 = None
    for c in state.players[P1_ID].battle_zone:
        if c.instance_id == 302:
            card_302 = c
            break
    assert card_302.is_tapped == False
    state.current_phase = dm_ai_module.Phase.ATTACK
    state.active_player_id = P1_ID
    attack_action = dm_ai_module.Action()
    attack_action.type = dm_ai_module.ActionType.ATTACK_PLAYER
    attack_action.source_instance_id = 301
    attack_action.target_player = P2_ID
    dm_ai_module.EffectResolver.resolve_action(state, attack_action, card_db)
    card_301 = None
    for c in state.players[P1_ID].battle_zone:
        if c.instance_id == 301:
            card_301 = c
            break
    assert card_301.is_tapped == True

def test_break_shield():
    state = setup_game()
    card_db = {}
    state.add_card_to_deck(P2_ID, 1, 401)
    dm_ai_module.DevTools.move_cards(state, P2_ID, dm_ai_module.Zone.DECK, dm_ai_module.Zone.SHIELD, 1)
    grave_def = dm_ai_module.ActionDef()
    grave_def.type = dm_ai_module.EffectActionType.SEND_SHIELD_TO_GRAVE
    state.active_player_id = P2_ID
    dm_ai_module.GenericCardSystem.resolve_action(state, grave_def, -1)
    assert len(state.players[P2_ID].shield_zone) == 0
    assert len(state.players[P2_ID].graveyard) == 1

def test_move_card_generic():
    state = setup_game()
    card_db = {}
    state.add_card_to_hand(P1_ID, 1, 501)
    dm_ai_module.DevTools.move_cards(state, P1_ID, dm_ai_module.Zone.HAND, dm_ai_module.Zone.GRAVEYARD, 1)
    assert len(state.players[P1_ID].hand) == 0
    assert len(state.players[P1_ID].graveyard) == 1

def test_condition_system():
    state = setup_game()
    state.active_player_id = 0
    cond = ConditionDef()
    cond.type = "DURING_YOUR_TURN"
    act = ActionDef()
    act.type = EffectActionType.DRAW_CARD
    act.value1 = 1
    eff = EffectDef()
    eff.trigger = TriggerType.NONE
    eff.condition = cond
    eff.actions = [act]
    state.add_card_to_deck(0, 1, 9001)
    prev_hand = len(state.players[0].hand)
    dm_ai_module.GenericCardSystem.resolve_effect(state, eff, 100)
    assert len(state.players[0].hand) == prev_hand + 1
    state.active_player_id = 1
    state.add_card_to_deck(0, 1, 9002)
    prev_hand = len(state.players[0].hand)
    dm_ai_module.GenericCardSystem.resolve_effect(state, eff, 100)
    assert len(state.players[0].hand) == prev_hand

def test_mana_armed():
    state = setup_game()
    state.active_player_id = 0
    card_db = {}
    # 8 arguments
    c1 = CardData(201, "Fire1", 1, "FIRE", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(c1)
    c2 = CardData(203, "Water1", 1, "WATER", 1000, "CREATURE", [], [])
    dm_ai_module.register_card_data(c2)
    cond = ConditionDef()
    cond.type = "MANA_ARMED"
    cond.value = 3
    cond.str_val = "FIRE"
    act = ActionDef()
    act.type = EffectActionType.DRAW_CARD
    act.value1 = 1
    eff = EffectDef()
    eff.trigger = TriggerType.NONE
    eff.condition = cond
    eff.actions = [act]
    state.add_card_to_mana(0, 201, 301)
    state.add_card_to_mana(0, 201, 302)
    state.add_card_to_mana(0, 203, 303)
    state.add_card_to_deck(0, 1, 9003)
    prev_hand = len(state.players[0].hand)
    dm_ai_module.GenericCardSystem.resolve_effect(state, eff, 301)
    assert len(state.players[0].hand) == prev_hand
    state.add_card_to_mana(0, 201, 304)
    prev_hand = len(state.players[0].hand)
    dm_ai_module.GenericCardSystem.resolve_effect(state, eff, 301)
    assert len(state.players[0].hand) == prev_hand + 1

def test_hyper_energy_cost_handler():
    state = setup_game()
    state.active_player_id = 0
    card_db = {}
    act = ActionDef()
    act.type = EffectActionType.COST_REFERENCE
    act.str_val = "FINISH_HYPER_ENERGY"
    act.value1 = 0
    eff = EffectDef()
    eff.trigger = TriggerType.NONE
    eff.actions = [act]
    state.add_test_card_to_battle(0, 1, 401, False, False)
    state.add_test_card_to_battle(0, 1, 402, False, False)
    targets = [401, 402]
    state.add_card_to_hand(0, 2, 300)
    dm_ai_module.GenericCardSystem.resolve_effect_with_targets(state, eff, targets, 300, card_db)
    c401 = None
    c402 = None
    for c in state.players[0].battle_zone:
        if c.instance_id == 401: c401 = c
        if c.instance_id == 402: c402 = c
    assert c401 is not None and c401.is_tapped
    assert c402 is not None and c402.is_tapped

if __name__ == "__main__":
    try:
        test_draw_card()
        test_mana_charge()
        test_tap_untap()
        test_break_shield()
        test_move_card_generic()
        test_condition_system()
        test_mana_armed()
        test_hyper_energy_cost_handler()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
