
import os
import sys
import random
import json
import time
import argparse
from typing import Dict, List, Optional
import numpy as np

# Ensure bin is in path (absolute path)
# Assuming this script is in python/training/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

# Import scenario definitions (reuse existing if possible or define here)
# For now, we redefine them to ensure we match the C++ config structure
SCENARIOS = {
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

class DataCollector:
    def __init__(self, card_db: Dict[int, dm_ai_module.CardDefinition]):
        self.card_db = card_db

    def create_instance(self, scenario_name: str) -> dm_ai_module.GameInstance:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_def = SCENARIOS[scenario_name]
        config_dict = scenario_def["config"]

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

        instance = dm_ai_module.GameInstance(42, self.card_db)
        instance.reset_with_scenario(config)
        return instance

    def run_episode(self, scenario_name: str, max_steps: int = 1000, debug: bool = False) -> Dict:
        instance = self.create_instance(scenario_name)
        state = instance.state

        step_count = 0
        start_time = time.time()

        while True:
            game_over, result = dm_ai_module.PhaseManager.check_game_over(state)
            if game_over:
                if debug:
                    print(f"Game Over! Result: {result}")
                return {
                    "result": int(result), # 1=P1_WIN, 2=P2_WIN, 3=DRAW
                    "steps": step_count,
                    "duration": time.time() - start_time,
                    "won": (int(result) == 1)
                }

            legal_actions = dm_ai_module.ActionGenerator.generate_legal_actions(state, self.card_db)

            if debug:
                print(f"Step {step_count}: Phase={state.current_phase}, ActivePlayer={state.active_player_id}")
                print(f"  Legal Actions ({len(legal_actions)}):")
                for action in legal_actions:
                    print(f"    Type: {action.type}, CardID: {action.card_id}, TargetInstance: {action.target_instance_id}")

            # If no legal actions, try to proceed to next phase
            if not legal_actions:
                if debug:
                    print("  No legal actions. Proceeding to next phase...")
                dm_ai_module.PhaseManager.next_phase(state, self.card_db)
                if debug:
                     print(f"  New Phase: {state.current_phase}")
                continue

            # Random Policy
            action = random.choice(legal_actions)
            if debug:
                print(f"  Chosen Action: {action.type}")

            prev_phase = state.current_phase
            dm_ai_module.EffectResolver.resolve_action(state, action, self.card_db)

            if action.type == dm_ai_module.ActionType.PASS:
                if state.current_phase == prev_phase:
                     dm_ai_module.PhaseManager.next_phase(state, self.card_db)

            step_count += 1

            if step_count > max_steps:
                 if debug:
                     print("Max steps reached.")
                 return {
                    "result": 3, # Treat as DRAW/TIMEOUT
                    "steps": step_count,
                    "duration": time.time() - start_time,
                    "won": False
                }

    def collect_data(self, scenario_name: str, episodes: int, output_file: str):
        print(f"Collecting data for scenario '{scenario_name}' ({episodes} episodes)...")
        results = []
        wins = 0

        # Run one debug episode first
        self.run_episode(scenario_name, debug=True)

        for i in range(episodes):
            res = self.run_episode(scenario_name)
            results.append(res)
            if res["won"]:
                wins += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{episodes}. Win Rate: {wins/(i+1):.2f}")

        win_rate = wins / episodes
        print(f"Finished. Total Win Rate: {win_rate:.2f}")

        # Save to file
        output_data = {
            "scenario": scenario_name,
            "episodes": episodes,
            "win_rate": win_rate,
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect training data from scenarios")
    parser.add_argument("--scenario", type=str, default="lethal_drill_easy", help="Scenario name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--output", type=str, default="training_data.json", help="Output file path")

    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print(f"Error: cards.json not found at {cards_path}")
        sys.exit(1)

    print(f"Loading cards from {cards_path}...")
    # Use dm_ai_module to load cards
    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    collector = DataCollector(card_db)
    collector.collect_data(args.scenario, args.episodes, args.output)
