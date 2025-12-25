
import os
import sys
import random
from typing import Dict, List, Optional, Any
import numpy as np

# Ensure bin is in path (absolute path)
bin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../bin'))
if bin_path not in sys.path:
    sys.path.append(bin_path)

import dm_ai_module

# Spec Step 2.2: Define combo practice board definitions in SCENARIOS
SCENARIOS: Dict[str, Dict[str, Any]] = {
    "infinite_loop_drill": {
        "description": "Start with a board state that allows an infinite loop. The goal is to prove the loop.",
        "config": {
            "my_mana": 5,
            "my_hand_cards": [1, 1], # Bronze-Arm Tribe
            "my_battle_zone": [2],   # Aqua Hulcus
            "my_mana_zone": [1, 1, 1, 1, 1],
            "my_grave_yard": [],
            "my_shields": [3], # Holy Awe
            "enemy_shield_count": 5,
            "enemy_battle_zone": [],
            "enemy_can_use_trigger": False,
            "loop_proof_mode": True # Hint to AI to prioritize loops? Or just a label.
        }
    },
    "lethal_drill_easy": {
        "description": "Enemy has 0 shields. Win this turn.",
        "config": {
            "my_mana": 3,
            "my_hand_cards": [1], # Bronze-Arm Tribe
            "my_battle_zone": [1], # Bronze-Arm Tribe
            "my_mana_zone": [1, 1, 1],
            "my_shields": [3],
            "enemy_shield_count": 0,
            "enemy_battle_zone": [],
            "enemy_can_use_trigger": False
        }
    }
}

class ScenarioRunner:
    def __init__(self, card_db: Dict[int, dm_ai_module.CardDefinition]) -> None:
        self.card_db = card_db
        self.executor = dm_ai_module.ScenarioExecutor(self.card_db)

    def get_config(self, scenario_name: str) -> dm_ai_module.ScenarioConfig:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_def = SCENARIOS[scenario_name]
        config_dict: Dict[str, Any] = scenario_def["config"]

        # Convert dict to ScenarioConfig
        # Note: dm_ai_module.ScenarioConfig has attributes like my_mana, etc.
        config = dm_ai_module.ScenarioConfig()
        config.my_mana = config_dict.get("my_mana", 0)
        config.my_hand_cards = config_dict.get("my_hand_cards", [])
        config.my_battle_zone = config_dict.get("my_battle_zone", [])
        config.my_mana_zone = config_dict.get("my_mana_zone", [])
        config.my_grave_yard = config_dict.get("my_grave_yard", [])
        config.my_shields = config_dict.get("my_shields", [])
        config.enemy_shield_count = config_dict.get("enemy_shield_count", 5)
        config.enemy_battle_zone = config_dict.get("enemy_battle_zone", [])
        config.enemy_can_use_trigger = config_dict.get("enemy_can_use_trigger", False)
        if "loop_proof_mode" in config_dict:
             config.loop_proof_mode = config_dict["loop_proof_mode"]
        return config

    def run_training_loop(self, scenario_name: str, episodes: int = 100) -> None:
        print(f"Starting training on scenario: {scenario_name}")
        success_count = 0

        config = self.get_config(scenario_name)

        for ep in range(episodes):
            # Run simulation entirely in C++ via Executor
            result_info = self.executor.run_scenario(config, 1000)

            if result_info.result == dm_ai_module.GameResult.P1_WIN:
                success_count += 1

            if (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{episodes} completed. Success rate: {success_count/(ep+1):.2f}")

        print(f"Training finished. Final success rate: {success_count/episodes:.2f}")

if __name__ == "__main__":
    # Load actual cards from JSON
    # Use absolute path to ensure it works regardless of where the script is run from
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    cards_path = os.path.join(project_root, 'data', 'cards.json')

    if not os.path.exists(cards_path):
        print(f"Error: cards.json not found at {cards_path}")
        sys.exit(1)

    print(f"Loading cards from {cards_path}...")
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)
    print(f"Loaded {len(card_db)} cards.")

    runner = ScenarioRunner(card_db)
    runner.run_training_loop("lethal_drill_easy", episodes=10)
