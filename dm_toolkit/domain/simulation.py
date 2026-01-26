# -*- coding: utf-8 -*-
import time
import gc
import os
from typing import Callable, Optional, List, Any, Dict
import numpy as np

# Import Backend Modules
from types import ModuleType

dm_ai_module: ModuleType | None
try:
    import dm_ai_module as _dm_ai_module  # type: ignore
    dm_ai_module = _dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.training.scenario_definitions import SCENARIOS
from dm_toolkit.gui.i18n import tr

class SimulationRunner:
    def __init__(self, card_db, scenario_name: str, episodes: int, threads: int, sims: int, evaluator_type: str, model_path: Optional[str] = None):
        self.card_db = card_db
        self.scenario_name = scenario_name
        self.episodes = episodes
        self.threads = threads
        self.sims = sims
        self.evaluator_type = evaluator_type # "Random", "Heuristic", "Model"
        self.model_path = model_path
        self.is_cancelled = False

        # Internal state
        self.neural_evaluator = None
        self.torch_network = None
        self.heuristic = None

    def cancel(self):
        self.is_cancelled = True

    def run(self, progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Runs the simulation.
        progress_callback(percent, message)
        Returns a summary dict.
        """
        def _report(p, m):
            if progress_callback:
                progress_callback(p, m)

        if not EngineCompat.is_available():
            raise RuntimeError(tr("Error: dm_ai_module not loaded."))

        assert dm_ai_module is not None

        _report(0, tr("Initializing..."))

        # Setup Scenario
        if self.scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario {self.scenario_name}")

        scenario_def = SCENARIOS[self.scenario_name]
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

        # Setup Evaluator
        evaluator_func = None

        try:
            if self.evaluator_type == "Model":
                try:
                    import torch
                    from dm_toolkit.ai.agent.network import AlphaZeroNetwork

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    # Input Size check
                    dummy_state = dm_ai_module.GameState(42)
                    dummy_vec = EngineCompat.TensorConverter_convert_to_tensor(dummy_state, 0, self.card_db)
                    input_size = len(dummy_vec)
                    action_size = 600

                    self.torch_network = AlphaZeroNetwork(input_size, action_size).to(device)

                    if self.model_path and os.path.exists(self.model_path):
                        self.torch_network.load_state_dict(torch.load(self.model_path, map_location=device))
                        _report(5, tr("Loaded model from {path}").format(path=self.model_path))
                    else:
                        _report(5, tr("Using initialized model (Untrained)"))

                    self.torch_network.eval()

                    # Batch Callback
                    def batch_inference(input_array):
                        with torch.no_grad():
                            tensor = torch.from_numpy(input_array).float().to(device)
                            policy_logits, values = self.torch_network(tensor)
                            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                            vals = values.squeeze(1).cpu().numpy()
                            return policies, vals

                    # Register global callback
                    EngineCompat.register_batch_inference_numpy(batch_inference)

                    self.neural_evaluator = dm_ai_module.NeuralEvaluator(self.card_db)
                    evaluator_func = self.neural_evaluator.evaluate

                except ImportError:
                    raise RuntimeError(tr("Error: PyTorch not available for Model evaluation."))
                except Exception as e:
                    raise RuntimeError(f"Error loading model: {e}")

            elif self.evaluator_type == "Heuristic":
                self.heuristic = dm_ai_module.HeuristicEvaluator(self.card_db)

                def heuristic_batch_evaluate(states):
                    results = []
                    for s in states:
                        p, v = self.heuristic.evaluate(s)
                        results.append((p, v))
                    return results

                evaluator_func = heuristic_batch_evaluate

            else: # Random
                def random_batch_evaluate(states):
                    results = []
                    for s in states:
                        policy = [1.0/600.0] * 600
                        value = 0.0
                        results.append((policy, value))
                    return results
                evaluator_func = random_batch_evaluate

            _report(10, tr("Starting simulation") + "...")

            # Simulation parameters
            batch_size = 32

            # Chunking Strategy for Large Scale Simulations
            chunk_size = 50 # Process 50 games at a time
            total_episodes = self.episodes
            num_chunks = (total_episodes + chunk_size - 1) // chunk_size

            all_results = []

            start_time = time.time()

            for chunk_idx in range(num_chunks):
                if self.is_cancelled:
                    _report(int((chunk_idx / num_chunks) * 90), tr("Simulation cancelled."))
                    break

                # Determine chunk range
                start_game_idx = chunk_idx * chunk_size
                end_game_idx = min((chunk_idx + 1) * chunk_size, total_episodes)
                current_chunk_size = end_game_idx - start_game_idx

                _report(
                    10 + int((chunk_idx / num_chunks) * 80),
                    tr("Processing chunk {idx}/{num} ({count} games)...").format(idx=chunk_idx + 1, num=num_chunks, count=current_chunk_size)
                )

                # Prepare Initial States for this chunk
                chunk_initial_states = []
                for i in range(current_chunk_size):
                    global_idx = start_game_idx + i
                    seed = int(time.time() * 1000 + global_idx) % 1000000
                    state = dm_ai_module.GameState(seed)
                    dm_ai_module.PhaseManager.setup_scenario(state, config, self.card_db)
                    chunk_initial_states.append(state)

                # Create Runner for this chunk
                runner = EngineCompat.create_parallel_runner(self.card_db, self.sims, batch_size)

                try:
                    results_info = EngineCompat.ParallelRunner_play_games(
                        runner, chunk_initial_states, evaluator_func,
                        temperature=1.0, add_noise=False, threads=self.threads
                    )
                    all_results.extend(results_info)
                finally:
                    # Clean up runner explicitly
                    del runner
                    gc.collect()

            duration = time.time() - start_time

            if self.is_cancelled and not all_results:
                 # Cancelled
                 return {
                     "status": "cancelled",
                     "all_results": all_results,
                     "duration": duration
                 }

            # Tally results
            wins = 0
            losses = 0
            draws = 0

            for info in all_results:
                if info.result == 1: wins += 1
                elif info.result == 2: losses += 1
                else: draws += 1

            total = wins + losses + draws
            win_rate = (wins / total * 100) if total > 0 else 0

            return {
                "status": "completed",
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "total": total,
                "win_rate": win_rate,
                "duration": duration,
                "throughput": total/duration if duration > 0 else 0
            }

        finally:
            # Cleanup Callback
            if self.evaluator_type == "Model":
                EngineCompat.register_batch_inference_numpy(None)
