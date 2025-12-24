
import os
import sys
import time
import argparse

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

from dm_toolkit.ai.agent.network import AlphaZeroNetwork
from dm_toolkit.ai.agent.transformer_model import DuelTransformer
from dm_toolkit.ai.agent.tokenization import game_state_to_tokens

# Import SCENARIOS
sys.path.append(os.path.dirname(__file__))
from scenario_definitions import SCENARIOS

import torch

class PerformanceVerifier:
    def __init__(self, card_db, model_path=None, model_type="resnet"):
        self.card_db = card_db
        self.model_type = model_type.lower()

        # Determine input size dynamically
        dummy_instance = dm_ai_module.GameInstance(42, self.card_db)
        dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_instance.state, 0, self.card_db)
        self.input_size = len(dummy_vec)
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if self.model_type == "transformer":
            print("Initializing TRANSFORMER model...")
            self.vocab_size = 1000 # Must match tokenization.py
            self.network = DuelTransformer(self.vocab_size, self.action_size).to(self.device)
        else:
            print("Initializing RESNET model...")
            self.network = AlphaZeroNetwork(self.input_size, self.action_size).to(self.device)

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            try:
                self.network.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model_name = os.path.basename(model_path)
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model_name = "LoadError"
        else:
            print("Using random initialized model (Baseline).")
            self.model_name = "Random/Untrained"

        self.network.eval()

        # Register Batch Callback
        # Based on bindings.cpp, we have set_flat_batch_callback for ResNet (State Vectors)
        # and set_sequence_batch_callback for Transformer (Tokens).
        # Currently, ParallelRunner uses NeuralEvaluator which likely calls "evaluate" which triggers the callback.
        # The NeuralEvaluator (C++) needs to know WHICH callback to trigger.
        # This is controlled by NeuralEvaluator::set_model_type.

        def batch_inference(input_array):
            # Input is numpy array (Batch, InputSize)
            with torch.no_grad():
                if self.model_type == "transformer":
                     # MOCK: Even with set_flat_batch_callback, if we are testing Transformer without C++ tokenization support,
                     # we receive floats. We mock tokenization here for pipeline verification.
                    B = input_array.shape[0]
                    dummy_tokens = torch.randint(0, self.vocab_size, (B, 32)).to(self.device)
                    policy_logits, values = self.network(dummy_tokens)
                else:
                    tensor = torch.from_numpy(input_array).float().to(self.device)
                    policy_logits, values = self.network(tensor)

                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                vals = values.squeeze(1).cpu().numpy()
                return policies, vals

        # We use set_flat_batch_callback because the C++ side currently sends state vectors (ParallelRunner default).
        # If we wanted true Transformer support, we'd need C++ to send tokens via set_sequence_batch_callback.
        if hasattr(dm_ai_module, "set_flat_batch_callback"):
            dm_ai_module.set_flat_batch_callback(batch_inference)
        elif hasattr(dm_ai_module, "register_batch_inference_numpy"):
             # Fallback for legacy binding name if existent
             dm_ai_module.register_batch_inference_numpy(batch_inference)
        else:
             print("Error: No batch callback registration function found in dm_ai_module.")
             sys.exit(1)

    def verify(self, scenario_name, episodes, mcts_sims=800, batch_size=32, num_threads=4):
        print(f"Verifying performance for '{self.model_name}' on scenario '{scenario_name}'...")
        print(f"Settings: sims={mcts_sims}, batch_size={batch_size}, threads={num_threads}")

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

        # Prepare initial states
        initial_states = []
        for i in range(episodes):
             instance = dm_ai_module.GameInstance(int(time.time() * 1000 + i) % 1000000, self.card_db)
             instance.reset_with_scenario(config)
             # Use clone() to ensure we have shared_ptr managed GameStates
             initial_states.append(instance.state.clone())

        # Setup ParallelRunner
        runner = dm_ai_module.ParallelRunner(self.card_db, mcts_sims, batch_size)
        neural_evaluator = dm_ai_module.NeuralEvaluator(self.card_db)

        start_time = time.time()

        # Run games
        # play_games returns vector<GameResultInfo>
        # Pass neural_evaluator object directly (instead of .evaluate method) to use optimized C++ path
        results_info = runner.play_games(
            initial_states,
            neural_evaluator,
            1.0,   # temperature
            False, # add_noise
            num_threads,
            0.0,   # alpha
            False  # collect_data
        )

        duration = time.time() - start_time
        if duration == 0: duration = 0.001

        # Aggregate Results
        stats = {
            "P1_WIN": 0,
            "P2_WIN": 0,
            "DRAW": 0
        }

        for info in results_info:
            res_val = info.result # enum int
            if res_val == 1: stats["P1_WIN"] += 1
            elif res_val == 2: stats["P2_WIN"] += 1
            else: stats["DRAW"] += 1

        total = episodes
        win_rate = stats["P1_WIN"] / total * 100

        print(f"--- Results for {self.model_name} ---")
        print(f"Total Games: {total}")
        print(f"Wins (P1): {stats['P1_WIN']} ({win_rate:.2f}%)")
        print(f"Losses (P2): {stats['P2_WIN']}")
        print(f"Draws: {stats['DRAW']}")
        print(f"Time Taken: {duration:.2f}s ({duration/total:.2f}s/game)")
        print(f"Throughput: {total / duration:.2f} games/s")

        return win_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="lethal_puzzle_easy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "transformer"], help="Model architecture type")

    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print("Error: cards.json not found")
        sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    verifier = PerformanceVerifier(card_db, args.model, model_type=args.model_type)
    try:
        verifier.verify(args.scenario, args.episodes, mcts_sims=args.sims, batch_size=args.batch_size, num_threads=args.threads)
    finally:
        # Cleanup to avoid Segfaults
        if hasattr(dm_ai_module, "clear_flat_batch_callback"):
            dm_ai_module.clear_flat_batch_callback()
        if hasattr(dm_ai_module, "clear_batch_inference_numpy"):
            dm_ai_module.clear_batch_inference_numpy()
