
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

# from dm_toolkit.ai.agent.network import AlphaZeroNetwork
from dm_toolkit.ai.agent.transformer_model import DuelTransformer

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
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Phase 8: Switch to Transformer
        self.network = DuelTransformer(self.input_size, self.action_size).to(self.device)

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model_name = os.path.basename(model_path)
        else:
            print("Using random initialized model (Baseline).")
            self.model_name = "Random/Untrained"

        self.network.eval()

        # Register Batch Callback
        def batch_inference(input_array):
            # Input array from C++ (numpy, float32)
            with torch.no_grad():
                tensor = torch.from_numpy(input_array).float().to(self.device)
                policy_logits, values = self.network(tensor)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                vals = values.squeeze(1).cpu().numpy()
                return policies, vals

        dm_ai_module.register_batch_inference_numpy(batch_inference)

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
             initial_states.append(instance.state)

        # Setup ParallelRunner
        # Note: NeuralEvaluator in Python doesn't have parallel logic itself, but C++ ParallelRunner does.
        # We need to pass a python wrapper around NeuralEvaluator.evaluate.
        # But wait, NeuralEvaluator is C++ binding.
        # dm_ai_module.NeuralEvaluator().evaluate is a python-callable that calls C++.

        # In ParallelRunner.play_games(..., evaluator, ...):
        # evaluator is std::function taking vector<GameState> returning pair<...>.
        # We can pass dm_ai_module.NeuralEvaluator(self.card_db).evaluate directly?
        # Yes, pybind11 should adapt it.

        runner = dm_ai_module.ParallelRunner(self.card_db, mcts_sims, batch_size)
        neural_evaluator = dm_ai_module.NeuralEvaluator(self.card_db)

        start_time = time.time()

        # Run games
        # play_games returns vector<GameResultInfo>
        # but binding for GameResultInfo might need inspection if we use it.
        # Actually binding returns list of GameResultInfo objects.
        # GameResultInfo has .result (int)

        # Pass collect_data=False to avoid memory accumulation (Memory Leak fix)
        results_info = runner.play_games(
            initial_states,
            neural_evaluator.evaluate,
            temperature=1.0,
            add_noise=False,
            num_threads=num_threads,
            alpha=0.0,
            collect_data=False
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

    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        print("Error: cards.json not found")
        sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    verifier = PerformanceVerifier(card_db, args.model)
    try:
        verifier.verify(args.scenario, args.episodes, mcts_sims=args.sims, batch_size=args.batch_size, num_threads=args.threads)
    finally:
        # Cleanup to avoid Segfaults
        if hasattr(dm_ai_module, "clear_batch_inference_numpy"):
            dm_ai_module.clear_batch_inference_numpy()
