import pytest
import sys
import os

# Add bin path to sys.path
bin_path = os.path.join(os.path.dirname(__file__), '..', 'bin')
sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    pytest.fail("dm_ai_module not found. Please build the C++ module first.")

def test_just_diver_attack():
    # Setup Card DB
    card_db = {
        1: dm_ai_module.CardDefinition(
            1, "Just Diver Creature", "WATER", ["Liquid People"], 2, 2000,
            dm_ai_module.CardKeywords(), []
        ),
        2: dm_ai_module.CardDefinition(
            2, "Enemy Attacker", "FIRE", ["Dragon"], 3, 5000,
            dm_ai_module.CardKeywords(), []
        )
    }

    # Enable Just Diver on card 1
    card_db[1].keywords.just_diver = True
    card_db[1].type = dm_ai_module.CardType.CREATURE
    # Enable Speed Attacker on card 2 to attack immediately
    card_db[2].keywords.speed_attacker = True
    card_db[2].type = dm_ai_module.CardType.CREATURE

    # Setup Game State
    game = dm_ai_module.GameState(100)
    game.turn_number = 1

    # Register CardData for Registry lookups (GenericCardSystem)
    for cid, cdef in card_db.items():
        cdata = dm_ai_module.CardData(cid, cdef.name, cdef.cost,
                                      cdef.civilization.name, # Use string name
                                      cdef.power if cdef.power > 0 else 0,
                                      "CREATURE" if cdef.type == dm_ai_module.CardType.CREATURE else "SPELL", cdef.races, [])
        dm_ai_module.register_card_data(cdata)

    # 1. Player 0 plays Just Diver Creature
    game.active_player_id = 0
    game.add_card_to_hand(0, 1, 100)

    # Cheat mana (Cost 2, Mana 3 to be safe)
    game.add_card_to_mana(0, 1, 900)
    game.add_card_to_mana(0, 1, 901)
    game.add_card_to_mana(0, 1, 902)

    # Ensure phase
    game.current_phase = dm_ai_module.Phase.MAIN

    # Play Flow
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
    play_action = next((a for a in actions if a.card_id == 1 and (a.type == dm_ai_module.ActionType.DECLARE_PLAY or a.type == dm_ai_module.ActionType.PLAY_CARD)), None)

    assert play_action is not None
    dm_ai_module.EffectResolver.resolve_action(game, play_action, card_db)

    if play_action.type == dm_ai_module.ActionType.DECLARE_PLAY:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        pay_action = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
        if pay_action: dm_ai_module.EffectResolver.resolve_action(game, pay_action, card_db)

        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        res_action = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
        if res_action: dm_ai_module.EffectResolver.resolve_action(game, res_action, card_db)

    # Verify turn_played
    p0_battle = game.players[0].battle_zone
    assert len(p0_battle) == 1
    jd_creature = p0_battle[0]

    # Re-setup with Speed Attacker to tap it via attack
    card_db[1].keywords.speed_attacker = True

    # Reset game
    game = dm_ai_module.GameState(100)
    game.turn_number = 1
    game.current_phase = dm_ai_module.Phase.MAIN

    # Play again
    game.active_player_id = 0
    game.add_card_to_hand(0, 1, 100)
    game.add_card_to_mana(0, 1, 900)
    game.add_card_to_mana(0, 1, 901)
    game.add_card_to_mana(0, 1, 902)

    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
    play_action = next((a for a in actions if a.card_id == 1 and (a.type == dm_ai_module.ActionType.DECLARE_PLAY or a.type == dm_ai_module.ActionType.PLAY_CARD)), None)
    dm_ai_module.EffectResolver.resolve_action(game, play_action, card_db)

    if play_action.type == dm_ai_module.ActionType.DECLARE_PLAY:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        pay_action = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
        if pay_action: dm_ai_module.EffectResolver.resolve_action(game, pay_action, card_db)
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        res_action = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
        if res_action: dm_ai_module.EffectResolver.resolve_action(game, res_action, card_db)

    # Attack with JD creature to tap it
    game.current_phase = dm_ai_module.Phase.ATTACK
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
    att_action = next((a for a in actions if a.type == dm_ai_module.ActionType.ATTACK_PLAYER), None)
    if att_action:
        dm_ai_module.EffectResolver.resolve_action(game, att_action, card_db)

    # Check tapped - using action generator state check implicitly later, or assume it worked.
    # Note: accessing .is_tapped on copy won't verify C++ state, but attacking should tap it.

    # 2. Switch to Opponent (Player 1)
    game.active_player_id = 1
    game.current_phase = dm_ai_module.Phase.MAIN

    # Add an attacker
    game.add_card_to_hand(1, 2, 200)
    for i in range(3): game.add_card_to_mana(1, 2, 902+i)
    # Extra mana
    game.add_card_to_mana(1, 2, 906)

    # Play Attacker
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
    play_attacker = next((a for a in actions if a.card_id == 2 and (a.type == dm_ai_module.ActionType.DECLARE_PLAY or a.type == dm_ai_module.ActionType.PLAY_CARD)), None)

    assert play_attacker is not None
    dm_ai_module.EffectResolver.resolve_action(game, play_attacker, card_db)

    if play_attacker.type == dm_ai_module.ActionType.DECLARE_PLAY:
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        pay_act = next((a for a in actions if a.type == dm_ai_module.ActionType.PAY_COST), None)
        if pay_act: dm_ai_module.EffectResolver.resolve_action(game, pay_act, card_db)
        actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
        res_act = next((a for a in actions if a.type == dm_ai_module.ActionType.RESOLVE_PLAY), None)
        if res_act: dm_ai_module.EffectResolver.resolve_action(game, res_act, card_db)

    # Move to Attack Phase
    game.current_phase = dm_ai_module.Phase.ATTACK

    # Generate Actions
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)

    # Expect: Attack Player (valid), Attack Creature (INVALID because Just Diver)

    can_attack_creature = False
    for a in actions:
        if a.type == dm_ai_module.ActionType.ATTACK_CREATURE:
            if a.target_instance_id == 100:
                can_attack_creature = True

    assert can_attack_creature == False, "Opponent should NOT be able to attack Just Diver creature"

    # 3. Fast Forward to expiry
    # P1 (T1) -> P0 (T2) [Expired] -> P1 (T2) [Expired, can attack]

    game.turn_number = 2
    game.active_player_id = 1

    # Simulate P0 Turn 2
    game.active_player_id = 0
    game.current_phase = dm_ai_module.Phase.ATTACK
    # Attack again
    actions = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)
    att_action = next((a for a in actions if a.type == dm_ai_module.ActionType.ATTACK_PLAYER), None)
    if att_action:
        dm_ai_module.EffectResolver.resolve_action(game, att_action, card_db)

    # T2 P1
    game.active_player_id = 1
    game.current_phase = dm_ai_module.Phase.ATTACK

    actions_t2 = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)

    can_attack_creature_t2 = False
    for a in actions_t2:
        if a.type == dm_ai_module.ActionType.ATTACK_CREATURE:
            if a.target_instance_id == 100:
                can_attack_creature_t2 = True

    assert can_attack_creature_t2 == True, "Opponent SHOULD be able to attack Just Diver creature after expiry"

if __name__ == "__main__":
    test_just_diver_attack()
