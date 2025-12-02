
import os
import sys
import numpy as np
import torch
import json
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

# Import SCENARIOS from scenario_definitions
sys.path.append(os.path.dirname(__file__))
from scenario_definitions import SCENARIOS

class TrainingDataCollector:
    def __init__(self, card_db, model_path=None):
        self.card_db = card_db

        # Determine input size dynamically
        dummy_instance = dm_ai_module.GameInstance(42, self.card_db)
        dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_instance.state, 0, self.card_db)
        self.input_size = len(dummy_vec)
        # Approximate upper bound from C++ constants (dm::ai::ActionEncoder::TOTAL_ACTION_SIZE)
        # Defined as ~600 in src/ai/encoders/action_encoder.hpp
        self.action_size = 600

        print(f"Network Config: Input Size={self.input_size}, Action Size={self.action_size}")

        self.network = AlphaZeroNetwork(self.input_size, self.action_size)
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.network.load_state_dict(torch.load(model_path))
        else:
            print("Using random initialized model.")

        self.network.eval()

    def run_episode(self, scenario_name, mcts_sims=50):
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
        state = instance.state

        mcts = MCTS(self.network, self.card_db, simulations=mcts_sims)

        training_examples = [] # (state_vec, policy, value_target)

        step_count = 0
        while True:
            is_over, result = dm_ai_module.PhaseManager.check_game_over(state)
            if is_over:
                return training_examples, result

            if step_count > 200:
                # Force draw
                return training_examples, 3

            # MCTS Search
            root = mcts.search(state, add_noise=(step_count < 10))

            # Extract Policy
            policy = np.zeros(self.action_size, dtype=np.float32)
            for child in root.children:
                if child.action:
                    idx = dm_ai_module.ActionEncoder.action_to_index(child.action)
                    if 0 <= idx < self.action_size:
                        policy[idx] = child.visit_count

            # Normalize policy
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy /= policy_sum
            else:
                # Should not happen if children exist, but just in case
                policy[:] = 1.0 / len(policy)

            # Store example
            # Note: We need to store the state vector relative to the current player
            state_vec = dm_ai_module.TensorConverter.convert_to_tensor(state, state.active_player_id, self.card_db)
            training_examples.append({
                "state": state_vec,
                "policy": policy,
                "player": state.active_player_id
            })

            # Select Action (Temperature = 1 for first k moves, then 0)
            # For simple training, let's just pick max visit count
            # or sample from policy

            # Re-normalize just to be super safe against float errors
            p_sum = np.sum(policy)
            if abs(p_sum - 1.0) > 1e-6:
                policy /= p_sum

            action_idx = np.random.choice(len(policy), p=policy)

            # Find the child corresponding to this index
            chosen_child = None
            for child in root.children:
                if child.action:
                    idx = dm_ai_module.ActionEncoder.action_to_index(child.action)
                    if idx == action_idx:
                        chosen_child = child
                        break

            if chosen_child is None:
                # Fallback (should not happen if policy consistent)
                if root.children:
                    chosen_child = root.children[0]
                else:
                    # No legal moves?
                    break

            # Advance state
            action = chosen_child.action
            dm_ai_module.EffectResolver.resolve_action(state, action, self.card_db)
            if action.type == dm_ai_module.ActionType.PASS:
                dm_ai_module.PhaseManager.next_phase(state, self.card_db)

            dm_ai_module.PhaseManager.fast_forward(state, self.card_db)
            step_count += 1

    def collect_data(self, scenario_name, episodes, output_file):
        print(f"Collecting data for scenario '{scenario_name}'...")
        all_data = [] # List of (state, policy, value)

        start_time = time.time()
        for i in range(episodes):
            examples, result = self.run_episode(scenario_name)

            # Assign value based on result
            # result: 1=P1_WIN, 2=P2_WIN, 3=DRAW
            for ex in examples:
                player = ex["player"]
                value = 0.0
                if result == 1: # P1 Wins
                    value = 1.0 if player == 0 else -1.0
                elif result == 2: # P2 Wins
                    value = 1.0 if player == 1 else -1.0
                elif result == 3: # Draw
                    value = 0.0

                all_data.append((ex["state"], ex["policy"], value))

            if (i+1) % 10 == 0:
                print(f"Collected {i+1}/{episodes} episodes. Total samples: {len(all_data)}")

        duration = time.time() - start_time
        print(f"Finished in {duration:.2f}s. Total samples: {len(all_data)}")

        # Save as npz
        states = np.array([x[0] for x in all_data], dtype=np.float32)
        policies = np.array([x[1] for x in all_data], dtype=np.float32)
        values = np.array([x[2] for x in all_data], dtype=np.float32)

        np.savez_compressed(output_file, states=states, policies=policies, values=values)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="lethal_puzzle_easy")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--output", type=str, default="training_data.npz")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        # Create dummy cards if not exists (should exist in this env)
        print("Warning: cards.json not found")
        sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    collector = TrainingDataCollector(card_db, args.model)
    collector.collect_data(args.scenario, args.episodes, args.output)
