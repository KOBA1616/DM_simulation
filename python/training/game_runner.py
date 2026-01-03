
import os
import sys
import numpy as np
import torch
import time
from typing import List, Tuple, Optional, Any, Callable, Dict, Union

# Ensure bin is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
bin_path = os.path.join(project_root, 'bin')
if bin_path not in sys.path:
    sys.path.append(bin_path)

try:
    import dm_ai_module
except ImportError:
    print(f"Error: Could not import dm_ai_module. Make sure it is built and in {bin_path}")
    sys.exit(1)

from dm_toolkit.ai.agent.network import AlphaZeroNetwork
from dm_toolkit.types import CardDB, GameState

class GameRunner:
    """
    Unified wrapper for running games using ParallelRunner.
    Handles initialization, execution, and result processing.
    """
    def __init__(self, card_db: CardDB, device: Optional[Any] = None) -> None:
        self.card_db: CardDB = card_db
        self.device: Any = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = dm_ai_module.TensorConverter.INPUT_SIZE
        self.action_size = dm_ai_module.ActionEncoder.TOTAL_ACTION_SIZE

    def create_network_callback(self, network: AlphaZeroNetwork) -> Callable[[List[Any]], Tuple[List[Any], List[float]]]:
        """
        Creates a callback function for ParallelRunner that uses the given network for inference.
        """
        def callback(states: List[Any]) -> Tuple[List[Any], List[float]]:
            if not states:
                return [], []

            try:
                # Flat batch conversion [B, InputSize]
                batch_flat = dm_ai_module.TensorConverter.convert_batch_flat(states, self.card_db, True)

                # Reshape to [B, InputSize]
                input_tensor = torch.tensor(batch_flat, dtype=torch.float32).view(len(states), self.input_size).to(self.device)

                with torch.no_grad():
                    policies_logits, values = network(input_tensor)

                policies = torch.softmax(policies_logits, dim=1).cpu().numpy().tolist()
                vals = values.cpu().numpy().flatten().tolist()

                return policies, vals
            except Exception as e:
                print(f"Error in network callback: {e}")
                return [], []

        return callback

    def prepare_initial_states(self, num_games: int, deck1: List[int], deck2: List[int]) -> List[GameState]:
        """
        Prepares initial GameState objects for a batch of games.
        """
        states = []
        for _ in range(num_games):
            state = dm_ai_module.GameState(len(self.card_db))
            state.set_deck(0, deck1)
            state.set_deck(1, deck2)
            state.initialize_card_stats(self.card_db, 40) # Assuming 40 card decks
            dm_ai_module.PhaseManager.start_game(state, self.card_db)
            states.append(state)
        return states

    def run_games(
        self,
        initial_states: List[GameState],
        evaluator_callback: Callable[[List[Any]], Tuple[List[Any], List[float]]],
        sims: int = 800,
        batch_size: int = 32,
        threads: int = 4,
        temperature: float = 1.0,
        add_noise: bool = True,
        collect_data: bool = False,
        pimc: bool = False,
        meta_decks_path: Optional[str] = None
    ) -> List[Any]: # Returns List[GameResultInfo]
        """
        Runs games using ParallelRunner.
        """
        runner = dm_ai_module.ParallelRunner(self.card_db, sims, batch_size)

        if pimc:
            runner.enable_pimc(True)
            if meta_decks_path:
                runner.load_meta_decks(meta_decks_path)

        # Set global callback if using the flat batch one
        if hasattr(dm_ai_module, 'set_flat_batch_callback'):
             # Wrap callback to match signature (states, player_id) -> (policies, values)
             def wrapped_callback(states: List[Any], player_id: int) -> Tuple[List[List[float]], List[float]]:
                 return evaluator_callback(states)

             dm_ai_module.set_flat_batch_callback(wrapped_callback)

        try:
            results = runner.play_games(
                initial_states,
                evaluator_callback,
                temperature,
                add_noise,
                threads,
                0.5, # alpha
                collect_data
            )
        finally:
            if hasattr(dm_ai_module, 'clear_flat_batch_callback'):
                dm_ai_module.clear_flat_batch_callback()

        return results

    def process_results_to_data(self, results: List[Any]) -> Dict[str, List[Any]]:
        """
        Extracts training data (states, policies, values) from game results.
        Converts GameState objects to feature tensors using TensorConverter.
        """
        data = {
            'states': [],
            'policies': [],
            'values': []
        }

        for res in results:
            if not res.states:
                continue

            winner = res.result

            # active_players tracks whose turn it was for each state
            # states list corresponds one-to-one with active_players

            # Batch convert states to tensors for efficiency if possible, or loop
            # TensorConverter.convert_batch_flat takes a list of states.
            # However, we need to respect the player perspective (state.active_player_id)
            # ParallelRunner ensures active_player_id is set on the state objects in the list.

            # Option 1: Convert one by one (safer for verification)
            # Option 2: Batch convert (faster)

            # Using one-by-one for robustness in this refactor, matching legacy behavior logic
            converted_states = []

            for state in res.states:
                # We need to convert based on the active player of that state
                pid = state.active_player_id
                # Convert to tensor (masked view)
                tensor_flat = dm_ai_module.TensorConverter.convert_to_tensor(state, pid, self.card_db, True)
                converted_states.append(tensor_flat)

            # Determine values
            step_values = []
            for player in res.active_players:
                if winner == dm_ai_module.GameResult.P1_WIN:
                    v = 1.0 if player == 0 else -1.0
                elif winner == dm_ai_module.GameResult.P2_WIN:
                    v = 1.0 if player == 1 else -1.0
                else:
                    v = 0.0
                step_values.append([v])

            data['states'].extend(converted_states)
            data['policies'].extend(res.policies)
            data['values'].extend(step_values)

        return data

    def save_data(self, data: Dict[str, List[Any]], output_path: str) -> None:
        """
        Saves processed data to .npz file.
        """
        if not data['states']:
            print("No data to save.")
            return

        print(f"Saving {len(data['states'])} samples to {output_path}...")
        np.savez_compressed(
            output_path,
            states=np.array(data['states'], dtype=np.float32),
            policies=np.array(data['policies'], dtype=np.float32),
            values=np.array(data['values'], dtype=np.float32)
        )
