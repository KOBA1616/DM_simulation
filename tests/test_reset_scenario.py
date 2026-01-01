import pytest
import os
import sys

# Add bin directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))

try:
    import dm_ai_module
except ImportError:
    pytest.skip("dm_ai_module not found", allow_module_level=True)

def test_reset_scenario():
    # Load card db
    db_path = os.path.join(os.path.dirname(__file__), '../data/cards.json')
    if not os.path.exists(db_path):
        pytest.skip("cards.json not found")

    card_db = dm_ai_module.JsonLoader.load_cards(db_path)

    # Create scenario config
    config = dm_ai_module.ScenarioConfig()
    config.my_hand_cards = [1, 2]
    config.my_battle_zone = [3]
    config.my_mana_zone = [4, 5]
    config.my_shields = [6]
    config.my_mana = 0
    config.enemy_shield_count = 5

    executor = dm_ai_module.ScenarioExecutor(card_db)
    # run_scenario(config, max_steps)
    executor.run_scenario(config, 100)
    print("Scenario ran successfully")

if __name__ == "__main__":
    test_reset_scenario()
