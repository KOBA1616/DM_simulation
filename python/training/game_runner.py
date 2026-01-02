import os
import sys
import numpy as np
import torch
from typing import List, Tuple, Any, Optional, Union, Dict

# Ensure paths if running as script or imported without setup
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
    pass # Assume caller handles this or it fails later

from dm_toolkit.ai.agent.network import AlphaZeroNetwork
from dm_toolkit.types import CardDB

class GameRunner:
    """
    Unified runner for executing games using C++ ParallelRunner with Python-side Neural Network inference.
    Replaces logic in self_play.py and train_pbt.py.
    """
    def __init__(self, card_db: CardDB, model_path: Optional[str] = None, device: Any = None):
        self.card_db = card_db
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network: Optional[AlphaZeroNetwork] = None
        self.model_path = model_path

        # Initialize dimensions using a dummy state
        dummy_state = dm_ai_module.GameState(len(self.card_db))
        dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_state, 0, self.card_db)
        self.input_size = len(dummy_vec)
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str) -> None:
        self.network = AlphaZeroNetwork(self.input_size, self.action_size).to(self.device)
        if path and os.path.exists(path):
            # print(f"Loading model: {path}")
            try:
                self.network.load_state_dict(torch.load(path, map_location=self.device))
            except Exception as e:
                print(f"Warning: Failed to load model {path}: {e}")
        self.network.eval()

    def _inference_callback(self, states: List[Any], *args) -> Tuple[List[List[float]], List[float]]:
        """
        Callback passed to C++ ParallelRunner.
        Receives batch of GameStates, converts to tensors, runs inference, returns policies and values.
        """
        if not states:
            return [], []

        try:
            # Use optimized C++ flat batch conversion
            # Returns a 1D list of floats
            batch_flat = dm_ai_module.TensorConverter.convert_batch_flat(states, self.card_db, True)

            # Convert to Tensor [B, InputSize]
            input_tensor = torch.tensor(batch_flat, dtype=torch.float32).view(len(states), self.input_size).to(self.device)

            with torch.no_grad():
                if self.network:
                    policies, values = self.network(input_tensor)
                    # Convert to standard Python lists for C++ binding
                    policies_list = policies.cpu().numpy().tolist()
                    values_list = values.cpu().numpy().flatten().tolist()
                    return policies_list, values_list
                else:
                    # Fallback if no network (shouldn't happen in typical usage)
                    # Return uniform policy and zero value
                    uniform = [1.0 / self.action_size] * self.action_size
                    return [uniform] * len(states), [0.0] * len(states)

        except Exception as e:
            print(f"Error in inference callback: {e}")
            return [], []

    def prepare_state(self, deck0: List[int], deck1: List[int]) -> Any:
        """Helper to create and initialize a GameState"""
        state = dm_ai_module.GameState(len(self.card_db))
        state.set_deck(0, deck0)
        state.set_deck(1, deck1)
        state.initialize_card_stats(self.card_db, 40)
        dm_ai_module.PhaseManager.start_game(state, self.card_db)
        return state

    def run_games(self, initial_states: List[Any], sims: int = 800, batch_size: int = 32, threads: int = 4,
                  temperature: float = 1.0, add_noise: bool = True, alpha: float = 0.5, collect_data: bool = False) -> List[Any]:
        """
        Execute games using ParallelRunner.
        """
        runner = dm_ai_module.ParallelRunner(self.card_db, sims, batch_size)

        # Call play_games
        # Using the signature that matches what train_pbt.py seemed to use (7 args)
        # or falling back if binding differs.
        try:
            return runner.play_games(
                initial_states,
                self._inference_callback,
                temperature,
                add_noise,
                threads,
                alpha,
                collect_data
            )
        except TypeError:
            # Fallback for legacy signature (5 args)
            return runner.play_games(
                initial_states,
                self._inference_callback,
                temperature,
                add_noise,
                threads
            )
