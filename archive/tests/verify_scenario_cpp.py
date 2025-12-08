
import os
import sys
import unittest

# Ensure bin is in path (absolute path)
bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../bin'))
if bin_path not in sys.path:
    sys.path.append(bin_path)

import dm_ai_module

class TestScenarioCpp(unittest.TestCase):
    def setUp(self):
        # Load card DB
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        cards_path = os.path.join(project_root, 'data', 'cards.json')
        if not os.path.exists(cards_path):
            self.fail(f"cards.json not found at {cards_path}")
        self.card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    def test_run_scenario_cpp_execution(self):
        # Create a config directly
        config = dm_ai_module.ScenarioConfig()
        config.my_mana = 3
        config.my_hand_cards = [1]
        config.my_mana_zone = [1, 1, 1, 1, 1]
        config.enemy_shield_count = 0

        # Use ScenarioExecutor
        executor = dm_ai_module.ScenarioExecutor(self.card_db)

        # Run scenario
        print("Running scenario via C++ ScenarioExecutor...")
        result_info = executor.run_scenario(config, 100)

        print(f"Scenario finished. Result: {result_info.result}, Turns: {result_info.turn_count}")

        # Check type
        self.assertIsInstance(result_info.result, dm_ai_module.GameResult)
        self.assertIsInstance(result_info.turn_count, int)

if __name__ == '__main__':
    unittest.main()
