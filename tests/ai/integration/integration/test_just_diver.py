
import sys
import os
import unittest
import pytest

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))

try:
    import dm_ai_module
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

def test_just_diver():
    # Setup Card DB
    card_db = {
        1: dm_ai_module.CardDefinition(
            1, "Just Diver Creature", "WATER", ["Liquid People"], 2, 2000,
            dm_ai_module.CardKeywords(), []
        ),
        2: dm_ai_module.CardDefinition(
            2, "Enemy Spell", "DARKNESS", ["Demon"], 3, 0,
            dm_ai_module.CardKeywords(), []
        ),
        3: dm_ai_module.CardDefinition(
            3, "Enemy Creature", "FIRE", ["Dragon"], 3, 3000,
            dm_ai_module.CardKeywords(), []
        )
    }

    # Enable Just Diver on card 1
    card_db[1].keywords.just_diver = True
    card_db[1].type = dm_ai_module.CardType.CREATURE
    card_db[2].type = dm_ai_module.CardType.SPELL
    card_db[3].type = dm_ai_module.CardType.CREATURE

    # Setup Game State
    game = dm_ai_module.GameState(100)
    game.active_player_id = 0
    game.turn_number = 1
    game.current_phase = dm_ai_module.Phase.MAIN

    # Register CardData for Registry lookups (GenericCardSystem)
    for cid, cdef in card_db.items():
        cdata = dm_ai_module.CardData(cid, cdef.name, cdef.cost,
                                      cdef.civilizations[0].name, # Use string name
                                      cdef.power if cdef.power > 0 else 0,
                                      "CREATURE" if cdef.type == dm_ai_module.CardType.CREATURE else "SPELL", cdef.races, [])
        dm_ai_module.register_card_data(cdata)

    # 1. Player 0 plays Just Diver Creature
    game.add_card_to_hand(0, 1, 100) # Player 0, Card 1, Instance 100

    # Cheat mana
    game.add_card_to_mana(0, 1, 900)
    game.add_card_to_mana(0, 1, 901) # 2 mana
    # Add EXTRA MANA to be safe (cost is 2, mana is 3)
    game.add_card_to_mana(0, 1, 902)

    # Generate PLAY action (DECLARE_PLAY)
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
    play_action = next((a for a in actions if a.card_id == 1 and (a.type == dm_ai_module.ActionType.DECLARE_PLAY or a.type == dm_ai_module.ActionType.PLAY_CARD)), None)

    assert play_action is not None, "Should be able to play Just Diver creature"

    dm_ai_module.EffectResolver.resolve_action(game, play_action, card_db)

    # If it was DECLARE_PLAY, we need to pay cost and resolve
    if play_action.type == dm_ai_module.ActionType.DECLARE_PLAY:
        # PAY_COST
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        pay_action = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
        if pay_action:
            dm_ai_module.EffectResolver.resolve_action(game, pay_action, card_db)

        # RESOLVE_PLAY
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        resolve_action = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
        if resolve_action:
            dm_ai_module.EffectResolver.resolve_action(game, resolve_action, card_db)

    # Check if it's in battle zone
    p1_battle = game.players[0].battle_zone
    assert len(p1_battle) == 1
    jd_creature = p1_battle[0]

    # Verify Just Diver Logic
    # Opponent (Player 1) tries to choose it as target.

    # Case A: Same Turn (Turn 1)
    # Setup Opponent (Player 1)
    game.active_player_id = 1 # Switch active player to opponent
    game.current_phase = dm_ai_module.Phase.MAIN

    # Opponent plays a spell that targets a creature
    effect_def = dm_ai_module.EffectDef()
    effect_def.trigger = dm_ai_module.TriggerType.ON_PLAY

    action_def = dm_ai_module.ActionDef()
    action_def.type = dm_ai_module.EffectActionType.DESTROY
    action_def.scope = dm_ai_module.TargetScope.TARGET_SELECT

    f_select = dm_ai_module.FilterDef()
    f_select.zones = ["BATTLE_ZONE"]
    f_select.owner = "OPPONENT"
    action_def.filter = f_select

    effect_def.actions = [action_def]

    # Register the spell card
    spell_id = 2
    spell_data = dm_ai_module.CardData(spell_id, "Destroyer", 3, "DARKNESS", 0, "SPELL", [], [effect_def])
    dm_ai_module.register_card_data(spell_data)

    # Update card_db
    card_db[spell_id] = dm_ai_module.CardDefinition(spell_id, "Destroyer", "DARKNESS", [], 3, 0, dm_ai_module.CardKeywords(), [])
    card_db[spell_id].type = dm_ai_module.CardType.SPELL

    # Play the spell
    game.add_card_to_hand(1, spell_id, 200) # Card 2 (Spell)
    game.add_card_to_mana(1, spell_id, 902)
    game.add_card_to_mana(1, spell_id, 903)
    game.add_card_to_mana(1, spell_id, 904)

    # Play flow for spell
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
    play_act = next((a for a in actions if a.card_id == spell_id and (a.type == dm_ai_module.ActionType.DECLARE_PLAY or a.type == dm_ai_module.ActionType.PLAY_CARD)), None)

    assert play_act is not None
    dm_ai_module.EffectResolver.resolve_action(game, play_act, card_db)

    if play_act.type == dm_ai_module.ActionType.DECLARE_PLAY:
        # PAY_COST
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        pay_act = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
        if pay_act:
             dm_ai_module.EffectResolver.resolve_action(game, pay_act, card_db)

        # RESOLVE_PLAY (triggers effect)
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        res_act = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
        if res_act:
             dm_ai_module.EffectResolver.resolve_action(game, res_act, card_db)

    # Now check legal actions. We expect SELECT_TARGET actions.
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)

    # We expect NO actions that target the JD creature (Instance 100).
    targets_jd = False
    for a in actions:
        if a.type == dm_ai_module.ActionType.SELECT_TARGET:
            if a.target_instance_id == 100:
                targets_jd = True

    assert targets_jd == False, "Opponent should NOT be able to target Just Diver creature on Turn 1"

    # Case B: Next Turn (Self Turn)
    game.active_player_id = 0
    game.turn_number = 2

    # We need to explicitly clear pending effects or reset state for the next check,
    # but simplest is to just verify state properties or start a fresh check.
    # The Just Diver flag is on the creature instance.
    # Let's try to target it again.

    # Let's setup a new game for Case B to be clean
    game2 = dm_ai_module.GameState(100)
    game2.active_player_id = 0
    game2.turn_number = 1
    game2.current_phase = dm_ai_module.Phase.MAIN

    # Play JD creature on Turn 1
    game2.add_card_to_hand(0, 1, 100)
    game2.add_card_to_mana(0, 1, 900)
    game2.add_card_to_mana(0, 1, 901)

    # Play
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game2, card_db)
    play_action = next((a for a in actions if a.card_id == 1 and (a.type == dm_ai_module.ActionType.DECLARE_PLAY or a.type == dm_ai_module.ActionType.PLAY_CARD)), None)
    assert play_action is not None, "Should be able to play Just Diver creature in game2"
    dm_ai_module.EffectResolver.resolve_action(game2, play_action, card_db)

    if play_action.type == dm_ai_module.ActionType.DECLARE_PLAY:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game2, card_db)
        pay_action = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
        if pay_action: dm_ai_module.EffectResolver.resolve_action(game2, pay_action, card_db)

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game2, card_db)
        res_action = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
        if res_action: dm_ai_module.EffectResolver.resolve_action(game2, res_action, card_db)

    # Advance to Turn 2
    game2.turn_number = 2
    game2.active_player_id = 1 # Opponent

    # Play Spell
    game2.add_card_to_hand(1, spell_id, 200)
    for i in range(3): game2.add_card_to_mana(1, spell_id, 910+i)

    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game2, card_db)
    play_act = next((a for a in actions if a.card_id == spell_id and (a.type == dm_ai_module.ActionType.DECLARE_PLAY or a.type == dm_ai_module.ActionType.PLAY_CARD)), None)
    assert play_act is not None, "Should be able to play spell in game2"
    dm_ai_module.EffectResolver.resolve_action(game2, play_act, card_db)

    if play_act.type == dm_ai_module.ActionType.DECLARE_PLAY:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game2, card_db)
        pay_act = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
        if pay_act: dm_ai_module.EffectResolver.resolve_action(game2, pay_act, card_db)

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game2, card_db)
        res_act = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
        if res_act: dm_ai_module.EffectResolver.resolve_action(game2, res_act, card_db)

    # Now check targets
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game2, card_db)

    targets_jd_t2 = False
    for a in actions:
        if a.type == dm_ai_module.ActionType.SELECT_TARGET:
            if a.target_instance_id == 100:
                targets_jd_t2 = True

    # On Turn 2 (Opponent turn), Just Diver should STILL protect the creature until START of my next turn (Turn 3).
    # So targets_jd_t2 should be False.
    assert targets_jd_t2 == False, "Opponent should NOT be able to target Just Diver creature on Turn 2 (before expiry)"

if __name__ == "__main__":
    test_just_diver()
