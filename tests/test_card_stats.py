
import sys
import os
import pytest
import dm_ai_module
from dm_ai_module import GameState, CardDefinition, CardType, Civilization, GameResult, ActionType, ActionGenerator

def create_mock_card_db():
    card_db = {}
    # Card 1: Bronze Arm Tribe (Creature, Cost 3)
    c1 = CardDefinition()
    c1.id = 1
    c1.name = "Bronze Arm Tribe"
    c1.type = CardType.CREATURE
    c1.civilization = Civilization.NATURE
    c1.cost = 3
    c1.power = 1000
    card_db[1] = c1

    # Card 2: Aqua Hulcus (Creature, Cost 3)
    c2 = CardDefinition()
    c2.id = 2
    c2.name = "Aqua Hulcus"
    c2.type = CardType.CREATURE
    c2.civilization = Civilization.WATER
    c2.cost = 3
    c2.power = 2000
    card_db[2] = c2
    return card_db

def test_card_stats_tracking():
    """
    Verifies that playing a card correctly updates its usage statistics in GameState.
    """
    gs = GameState(42)
    card_db = create_mock_card_db()

    # Initialize stats for all cards in DB
    dm_ai_module.initialize_card_stats(gs, card_db, 40)

    gi = dm_ai_module.GameInstance(42, card_db)
    config = dm_ai_module.ScenarioConfig()
    config.my_hand_cards = [1] # Card 1 in hand
    config.my_mana_zone = [1] * 3 # 3 Mana
    gi.reset_with_scenario(config)

    state = gi.state
    state.current_phase = dm_ai_module.Phase.MAIN

    # Initialize stats on this new state
    dm_ai_module.initialize_card_stats(state, card_db, 40)

    # Generate play action
    actions = ActionGenerator.generate_legal_actions(state, card_db)
    play_action = None
    for action in actions:
        if action.type == ActionType.PLAY_CARD and action.card_id == 1:
            play_action = action
            break

    assert play_action is not None

    # Execute action
    dm_ai_module.EffectResolver.resolve_action(state, play_action, card_db)

    # Check stats again
    stats_after = dm_ai_module.get_card_stats(state)
    assert 1 in stats_after
    assert stats_after[1]['play_count'] == 1
    assert stats_after[1]['sum_cost_discount'] == 0 # Cost 3 played with 3 mana

    # Check early usage
    # Scenario defaults to Turn 5.
    assert state.turn_number == 5
    # So early_usage should be 0.
    assert stats_after[1]['sum_early_usage'] == 0.0

def test_card_stats_win_contribution():
    """
    Verifies that winning a game updates win contribution stats.
    """
    card_db = create_mock_card_db()
    gi = dm_ai_module.GameInstance(42, card_db)

    config = dm_ai_module.ScenarioConfig()
    config.my_hand_cards = [2] # Aqua Hulcus
    config.my_mana_zone = [2] * 3
    config.my_battle_zone = [1] # Bronze Arm Tribe (attacker)
    config.enemy_shield_count = 0
    gi.reset_with_scenario(config)

    state = gi.state
    # Initialize stats
    dm_ai_module.initialize_card_stats(state, card_db, 40)

    # Play Aqua Hulcus (Card 2)
    state.current_phase = dm_ai_module.Phase.MAIN
    actions = ActionGenerator.generate_legal_actions(state, card_db)
    play_action = [a for a in actions if a.type == ActionType.PLAY_CARD and a.card_id == 2][0]
    dm_ai_module.EffectResolver.resolve_action(state, play_action, card_db)

    # Proceed to Attack Phase
    state.current_phase = dm_ai_module.Phase.ATTACK

    # Attack Player with Card 1
    actions = ActionGenerator.generate_legal_actions(state, card_db)
    attack_action = None
    for a in actions:
        if a.type == ActionType.ATTACK_PLAYER:
            attack_action = a
            break

    assert attack_action is not None

    dm_ai_module.EffectResolver.resolve_action(state, attack_action, card_db)

    # Block Phase -> Pass
    state.current_phase = dm_ai_module.Phase.BLOCK
    actions = ActionGenerator.generate_legal_actions(state, card_db)
    pass_action = [a for a in actions if a.type == ActionType.PASS][0]
    dm_ai_module.EffectResolver.resolve_action(state, pass_action, card_db)

    # Trigger game over check via fast_forward
    dm_ai_module.PhaseManager.fast_forward(state, card_db)

    assert state.winner == dm_ai_module.GameResult.P1_WIN

    # Check stats for Card 2
    stats = dm_ai_module.get_card_stats(state)
    assert 2 in stats
    assert stats[2]['play_count'] == 1
    assert stats[2]['win_count'] == 1
    assert stats[2]['sum_win_contribution'] == 1.0
