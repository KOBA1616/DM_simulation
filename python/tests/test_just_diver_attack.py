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
    # Enable Speed Attacker on card 2 to attack immediately
    card_db[2].keywords.speed_attacker = True

    # Setup Game State
    game = dm_ai_module.GameState(100)
    game.turn_number = 1

    # 1. Player 0 plays Just Diver Creature
    game.active_player_id = 0
    game.add_card_to_hand(0, 1, 100)

    # Cheat mana
    game.add_card_to_mana(0, 1, 900)
    game.add_card_to_mana(0, 1, 901)

    action_play = dm_ai_module.Action()
    action_play.type = dm_ai_module.ActionType.PLAY_CARD
    action_play.source_instance_id = 100
    action_play.target_player = 0

    dm_ai_module.EffectResolver.resolve_action(game, action_play, card_db)

    # Verify turn_played
    p0_battle = game.players[0].battle_zone
    assert len(p0_battle) == 1
    jd_creature = p0_battle[0]

    print(f"Turn Played: {jd_creature.turn_played}, Game Turn: {game.turn_number}")
    # assert jd_creature.turn_played == 1, f"Expected turn_played=1, got {jd_creature.turn_played}"

    # Tap the Just Diver creature to make it a valid attack target normally
    p0_battle[0].is_tapped = True

    # 2. Switch to Opponent (Player 1)
    game.active_player_id = 1

    # Add an attacker
    game.add_card_to_hand(1, 2, 200)
    game.add_card_to_mana(1, 2, 902)
    game.add_card_to_mana(1, 2, 903)
    game.add_card_to_mana(1, 2, 904)

    play_attacker = dm_ai_module.Action()
    play_attacker.type = dm_ai_module.ActionType.PLAY_CARD
    play_attacker.source_instance_id = 200
    play_attacker.target_player = 1

    dm_ai_module.EffectResolver.resolve_action(game, play_attacker, card_db)

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

    # Make sure we still have the attacker and target is tapped
    assert len(game.players[1].battle_zone) == 1
    assert len(game.players[0].battle_zone) == 1
    game.players[0].battle_zone[0].is_tapped = True

    actions_t2 = dm_ai_module.ActionGenerator.generate_legal_actions(game, card_db)

    can_attack_creature_t2 = False
    for a in actions_t2:
        if a.type == dm_ai_module.ActionType.ATTACK_CREATURE:
            if a.target_instance_id == 100:
                can_attack_creature_t2 = True

    assert can_attack_creature_t2 == True, "Opponent SHOULD be able to attack Just Diver creature after expiry"

if __name__ == "__main__":
    test_just_diver_attack()
