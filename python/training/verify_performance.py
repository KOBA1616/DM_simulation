
import os
import sys
import numpy as np
import torch
import argparse
import time

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from py_ai.agent.network import AlphaZeroNetwork
from py_ai.agent.mcts import MCTS

# Import SCENARIOS
sys.path.append(os.path.dirname(__file__))
from scenario_definitions import SCENARIOS

class PerformanceVerifier:
    def __init__(self, card_db, model_path=None):
        self.card_db = card_db

        # Determine input size dynamically
        dummy_instance = dm_ai_module.GameInstance(42, self.card_db)
        dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_instance.state, 0, self.card_db)
        self.input_size = len(dummy_vec)
        self.action_size = 600

        self.network = AlphaZeroNetwork(self.input_size, self.action_size)

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.network.load_state_dict(torch.load(model_path))
            self.model_name = os.path.basename(model_path)
        else:
            print("Using random initialized model (Baseline).")
            self.model_name = "Random/Untrained"

        self.network.eval()

    def run_episode(self, scenario_name, mcts_sims=800):
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

        # Create game instance
        # Seed is random for verification to test robustness, unless fixed seed requested
        instance = dm_ai_module.GameInstance(int(time.time() * 1000) % 1000000, self.card_db)
        instance.reset_with_scenario(config)
        state = instance.state

        mcts = MCTS(self.network, self.card_db, simulations=mcts_sims)

        step_count = 0
        max_steps = 200

        while True:
            is_over, result = dm_ai_module.PhaseManager.check_game_over(state)
            if is_over:
                return result # 1=P1, 2=P2, 3=Draw

            if step_count > max_steps:
                return 3 # Force Draw

            # MCTS Search
            # No noise for evaluation to be deterministic(ish)
            root = mcts.search(state, add_noise=False)

            # Select Best Action (Greedy)
            best_action = None
            max_visits = -1

            for child in root.children:
                if child.visit_count > max_visits:
                    max_visits = child.visit_count
                    best_action = child.action

            if best_action is None:
                # Fallback if no children (should not happen usually)
                if root.children:
                    best_action = root.children[0].action
                else:
                    break

            # Advance state
            dm_ai_module.EffectResolver.resolve_action(state, best_action, self.card_db)
            if best_action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(state, self.card_db)

            dm_ai_module.PhaseManager.fast_forward(state, self.card_db)
            step_count += 1

    def verify(self, scenario_name, episodes):
        print(f"Verifying performance for '{self.model_name}' on scenario '{scenario_name}'...")

        results = {
            "P1_WIN": 0,
            "P2_WIN": 0,
            "DRAW": 0
        }

        start_time = time.time()
        for i in range(episodes):
            res = self.run_episode(scenario_name)
            if res == 1:
                results["P1_WIN"] += 1
            elif res == 2:
                results["P2_WIN"] += 1
            else:
                results["DRAW"] += 1

            if (i+1) % 10 == 0:
                print(f"Played {i+1}/{episodes}...")

        duration = time.time() - start_time

        total = episodes
        win_rate = results["P1_WIN"] / total * 100

        print(f"--- Results for {self.model_name} ---")
        print(f"Total Games: {total}")
        print(f"Wins (P1): {results['P1_WIN']} ({win_rate:.2f}%)")
        print(f"Losses (P2): {results['P2_WIN']}")
        print(f"Draws: {results['DRAW']}")
        print(f"Time Taken: {duration:.2f}s")

        return win_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="lethal_puzzle_easy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print("Error: cards.json not found")
        sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    verifier = PerformanceVerifier(card_db, args.model)
    verifier.verify(args.scenario, args.episodes)
