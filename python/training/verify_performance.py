
import os
import sys
import time
import argparse
import json
import numpy as np

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
python_path = os.path.join(project_root, 'python')

if bin_path not in sys.path:
    sys.path.append(bin_path)
if python_path not in sys.path:
    sys.path.append(python_path)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from dm_toolkit.ai.agent.network import AlphaZeroNetwork, AlphaZeroTransformer
from typing import Any, List, Tuple, Optional

# Import SCENARIOS
try:
    from dm_toolkit.training.scenario_definitions import SCENARIOS
except ImportError:
    # Fallback or try adding python path if running directly
    sys.path.append(os.path.join(project_root, 'python'))
    try:
        from training.scenario_definitions import SCENARIOS
    except ImportError:
         # Try dm_toolkit directly
         from dm_toolkit.training.scenario_definitions import SCENARIOS

import torch

class PerformanceVerifier:
    def __init__(self, card_db: Any, model_path: Optional[str] = None, model_type: str = "resnet") -> None:
        self.card_db: Any = card_db
        self.model_type: str = model_type.lower()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Determine input size dynamically for ResNet
        dummy_instance = dm_ai_module.GameInstance(42, self.card_db)
        # Note: TensorConverter might require specific bindings, ensuring robust access
        if hasattr(dm_ai_module.TensorConverter, "convert_to_tensor"):
            dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_instance.state, 0, self.card_db)
            self.input_size = len(dummy_vec)
        else:
            self.input_size = 205 # Fallback if direct access fails

        # Initialize Network based on type
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

        # Initialize Network based on type
        self.network: Any
        if self.model_type == "transformer":
            print("Initializing TRANSFORMER model (AlphaZeroTransformer)...")
            self.vocab_size = dm_ai_module.TokenConverter.get_vocab_size()
            self.max_seq_len = 200 # Should match training
            self.network = AlphaZeroTransformer(action_size=self.action_size,
                                                vocab_size=self.vocab_size,
                                                max_seq_len=self.max_seq_len).to(self.device)
        else:
            print("Initializing RESNET model (AlphaZeroNetwork)...")
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
        self._register_callback()

    def _register_callback(self) -> None:
        """Registers the appropriate C++ callback for batch inference."""

        if self.model_type == "transformer":
            def sequence_batch_inference(token_lists: List[List[int]]) -> Tuple[List[List[float]], List[float]]:
                # token_lists is List[List[int]] from C++
                # Convert to padded tensor
                # We need to manually pad here because C++ sends raw vectors

                max_len = 0
                for tokens in token_lists:
                    max_len = max(max_len, len(tokens))

                # Ensure min length if needed, though NetworkV2 handles it
                max_len = max(max_len, 1)

                batch_size = len(token_lists)
                padded_tensor = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)
                mask_tensor = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device)

                for i, tokens in enumerate(token_lists):
                    L = len(tokens)
                    if L > 0:
                        t = torch.tensor(tokens, dtype=torch.long, device=self.device)
                        padded_tensor[i, :L] = t
                        mask_tensor[i, :L] = True

                with torch.no_grad():
                    policy_logits, values = self.network(padded_tensor, mask=mask_tensor)

                    policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                    vals = values.squeeze(1).cpu().numpy()

                    # Return tuple of list-of-lists and list-of-floats
                    # C++ expects vector<vector<float>> policies, vector<float> values
                    return policies.tolist(), vals.tolist()

            if hasattr(dm_ai_module, "set_sequence_batch_callback"):
                dm_ai_module.set_sequence_batch_callback(sequence_batch_inference)
                print("Registered sequence batch callback.")
            else:
                print("Error: set_sequence_batch_callback not found in module.")
                sys.exit(1)

        else:
            # ResNet / Flat Tensor
            def flat_batch_inference(input_array: Any, *args: Any) -> Tuple[Any, Any]:
                # Input is numpy array (Batch, InputSize) (via buffer protocol/copy in binding)
                # Accepts *args to handle additional arguments passed by C++ (e.g. valid_action_masks) which are ignored for now.
                # The binding `dm::python::call_flat_batch_callback` converts C++ vector to numpy array.

                with torch.no_grad():
                    # Input array comes as a single flat float32 array or list of lists depending on binding implementation.
                    # Based on standard usage in this project:
                    # It seems to receive a numpy array or a list of lists.

                    if isinstance(input_array, list):
                         input_array = np.array(input_array, dtype=np.float32)
                    
                    # Reshape if flat [Batch * InputSize]
                    if input_array.ndim == 1:
                         # Calculate batch size dynamically or use -1 if InputSize is known
                         if self.input_size > 0:
                             input_array = input_array.reshape(-1, self.input_size)

                    tensor = torch.from_numpy(input_array).float().to(self.device)
                    policy_logits, values = self.network(tensor)

                    policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                    vals = values.squeeze(1).cpu().numpy()
                    return policies, vals

            if hasattr(dm_ai_module, "set_flat_batch_callback"):
                dm_ai_module.set_flat_batch_callback(flat_batch_inference)
                print("Registered flat batch callback.")
            else:
                 print("Error: No batch callback registration function found in dm_ai_module.")
                 sys.exit(1)

    def verify(self, scenario_name: str, episodes: int, mcts_sims: int = 800, batch_size: int = 32, num_threads: int = 4,
               pimc: bool = False, meta_decks_path: Optional[str] = None) -> float:
        print(f"Verifying performance for '{self.model_name}' on scenario '{scenario_name}'...")
        print(f"Settings: sims={mcts_sims}, batch_size={batch_size}, threads={num_threads}, mode={self.model_type}, pimc={pimc}")

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

        if pimc:
            runner.enable_pimc(True)
            if meta_decks_path:
                 runner.load_meta_decks(meta_decks_path)
                 print(f"Loaded meta decks from {meta_decks_path}")

        neural_evaluator = dm_ai_module.NeuralEvaluator(self.card_db)

        # IMPORTANT: Set the model type on the C++ evaluator
        if self.model_type == "transformer":
            neural_evaluator.set_model_type(dm_ai_module.ModelType.TRANSFORMER)
        else:
            neural_evaluator.set_model_type(dm_ai_module.ModelType.RESNET)

        start_time = time.time()

        # Run games
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
    parser.add_argument("--result_file", type=str, help="Path to save JSON result")
    parser.add_argument("--scenario", type=str, default="lethal_puzzle_easy")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--model_type", type=str, default="resnet", choices=["resnet", "transformer"], help="Model architecture type")
    parser.add_argument("--pimc", action="store_true", help="Enable PIMC (Perfect Information Monte Carlo)")
    parser.add_argument("--meta_decks", type=str, default="data/meta_decks.json", help="Path to meta decks JSON for PIMC")

    args = parser.parse_args()

    cards_path = os.path.join(project_root, 'data', 'cards.json')
    if not os.path.exists(cards_path):
        # Try finding it relative to current working directory if project_root logic fails or is different
        cards_path = os.path.abspath('data/cards.json')
        if not os.path.exists(cards_path):
            print(f"Error: cards.json not found at {cards_path}")
            sys.exit(1)

    card_db = dm_ai_module.JsonLoader.load_cards(cards_path)

    verifier = PerformanceVerifier(card_db, args.model, model_type=args.model_type)
    try:
        win_rate = verifier.verify(args.scenario, args.episodes, mcts_sims=args.sims, batch_size=args.batch_size, num_threads=args.threads,
                        pimc=args.pimc, meta_decks_path=args.meta_decks)

        if args.result_file:
            with open(args.result_file, 'w', encoding='utf-8') as f:
                json.dump({"win_rate": win_rate}, f, ensure_ascii=False)
            print(f"Result saved to {args.result_file}")
    finally:
        # Cleanup to avoid Segfaults
        if hasattr(dm_ai_module, "clear_flat_batch_callback"):
            dm_ai_module.clear_flat_batch_callback()
        if hasattr(dm_ai_module, "clear_sequence_batch_callback"):
            dm_ai_module.clear_sequence_batch_callback()
