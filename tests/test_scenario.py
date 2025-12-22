import sys
import os
import pytest

# Add the bin directory to sys.path
sys.path.append(os.path.join(os.getcwd(), 'bin'))

try:
    import dm_ai_module
except ImportError:
    pass

def test_scenario_config():
    config = dm_ai_module.ScenarioConfig()
    config.my_mana = 5
    config.my_hand_cards = [1, 2, 3]
    config.enemy_shield_count = 2

    assert config.my_mana == 5
    assert config.my_hand_cards == [1, 2, 3]
    assert config.enemy_shield_count == 2

def test_game_instance_scenario():
    # Setup card DB
    card_db = dm_ai_module.JsonLoader.load_cards("data/cards.json")

    gi = dm_ai_module.GameInstance(42, card_db)

    config = dm_ai_module.ScenarioConfig()
    config.my_mana = 3
    # Use IDs that likely exist if loaded, else standard ones
    config.my_hand_cards = [1, 1]
    config.my_battle_zone = [2]
    config.my_mana_zone = [1, 2, 3]
    config.enemy_shield_count = 4

    gi.reset_with_scenario(config)

    state = gi.state
    assert state.turn_number == 5

    # Check player 0 resources
    p0 = state.players[0]
    assert len(p0.hand) == 2
    assert p0.hand[0].card_id == 1
    assert len(p0.battle_zone) == 1
    assert p0.battle_zone[0].card_id == 2
    assert len(p0.mana_zone) == 3

    # Check player 1 resources
    p1 = state.players[1]
    assert len(p1.shield_zone) == 4

    print("Scenario test passed!")

if __name__ == "__main__":
    test_scenario_config()
    test_game_instance_scenario()
