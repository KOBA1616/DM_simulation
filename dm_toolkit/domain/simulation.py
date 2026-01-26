# dm_toolkit/domain/simulation.py
import time
import gc
import os
from typing import Callable, Optional, Any, List
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.training.scenario_definitions import SCENARIOS

# Type hints
try:
    import dm_ai_module as _dm_ai_module
    dm_ai_module = _dm_ai_module
except ImportError:
    dm_ai_module = None

class SimulationRunner:
    def __init__(self, card_db, scenario_name: str, episodes: int, threads: int, sims: int, evaluator_type: str, model_path: str = None):
        self.card_db = card_db
        self.scenario_name = scenario_name
        self.episodes = episodes
        self.threads = threads
        self.sims = sims
        self.evaluator_type = evaluator_type
        self.model_path = model_path
        self._cancelled = False
        self.engine_compat = EngineCompat

    def cancel(self):
        self._cancelled = True

    def run(self,
            progress_callback: Callable[[int, str], None],
            finished_callback: Callable[[float, str], None],
            tr: Callable[[str], str] = lambda s: s):

        if not self.engine_compat.is_available():
            finished_callback(0.0, tr("Error: dm_ai_module not loaded."))
            return

        assert dm_ai_module is not None

        progress_callback(0, tr("Initializing..."))

        # Setup Scenario
        if self.scenario_name not in SCENARIOS:
            finished_callback(0.0, f"Error: Unknown scenario {self.scenario_name}")
            return

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
        self.neural_evaluator = None
        self.torch_network = None

        if self.evaluator_type == "Model":
            try:
                import torch
                from dm_toolkit.ai.agent.network import AlphaZeroNetwork

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Input Size check
                dummy_state = dm_ai_module.GameState(42)
                dummy_vec = self.engine_compat.TensorConverter_convert_to_tensor(dummy_state, 0, self.card_db)
                input_size = len(dummy_vec)
                action_size = 600

                self.torch_network = AlphaZeroNetwork(input_size, action_size).to(device)

                if self.model_path and os.path.exists(self.model_path):
                    self.torch_network.load_state_dict(torch.load(self.model_path, map_location=device))
                    progress_callback(5, tr("Loaded model from {path}").format(path=self.model_path))
                else:
                    progress_callback(5, tr("Using initialized model (Untrained)"))

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
                self.engine_compat.register_batch_inference_numpy(batch_inference)

                self.neural_evaluator = dm_ai_module.NeuralEvaluator(self.card_db)
                evaluator_func = self.neural_evaluator.evaluate

            except ImportError:
                finished_callback(0.0, tr("Error: PyTorch not available for Model evaluation."))
                return
            except Exception as e:
                finished_callback(0.0, f"Error loading model: {e}")
                return

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

        progress_callback(10, tr("Starting simulation") + "...")

        # Simulation parameters
        batch_size = 32
        chunk_size = 50
        total_episodes = self.episodes
        num_chunks = (total_episodes + chunk_size - 1) // chunk_size

        all_results = []
        start_time = time.time()

        try:
            for chunk_idx in range(num_chunks):
                if self._cancelled:
                    progress_callback(int((chunk_idx / num_chunks) * 90), tr("Simulation cancelled."))
                    break

                start_game_idx = chunk_idx * chunk_size
                end_game_idx = min((chunk_idx + 1) * chunk_size, total_episodes)
                current_chunk_size = end_game_idx - start_game_idx

                progress_callback(
                    10 + int((chunk_idx / num_chunks) * 80),
                    tr("Processing chunk {idx}/{num} ({count} games)...").format(idx=chunk_idx + 1, num=num_chunks, count=current_chunk_size)
                )

                chunk_initial_states = []
                for i in range(current_chunk_size):
                    global_idx = start_game_idx + i
                    seed = int(time.time() * 1000 + global_idx) % 1000000
                    state = dm_ai_module.GameState(seed)
                    dm_ai_module.PhaseManager.setup_scenario(state, config, self.card_db)
                    chunk_initial_states.append(state)

                runner = self.engine_compat.create_parallel_runner(self.card_db, self.sims, batch_size)

                try:
                    results_info = self.engine_compat.ParallelRunner_play_games(
                        runner, chunk_initial_states, evaluator_func,
                        temperature=1.0, add_noise=False, threads=self.threads
                    )
                    all_results.extend(results_info)
                except Exception as e:
                    finished_callback(0.0, f"{tr('Simulation Error')} in chunk {chunk_idx}: {e}")
                    return
                finally:
                    del runner
                    gc.collect()

            if self._cancelled and not all_results:
                 finished_callback(0.0, tr("Simulation cancelled by user."))
                 return

            duration = time.time() - start_time
            wins = 0
            losses = 0
            draws = 0

            for info in all_results:
                if info.result == 1: wins += 1
                elif info.result == 2: losses += 1
                else: draws += 1

            total = wins + losses + draws
            win_rate = (wins / total * 100) if total > 0 else 0

            summary = (
                f"{tr('Completed')} {total} episodes in {duration:.2f}s.\n"
                f"{tr('Wins')}: {wins} ({win_rate:.1f}%)\n"
                f"{tr('Losses')}: {losses}\n"
                f"{tr('Draws')}: {draws}\n"
                f"{tr('Throughput')}: {total/duration:.1f} games/s"
            )

            finished_callback(win_rate, summary)

        finally:
            if self.evaluator_type == "Model":
                self.engine_compat.register_batch_inference_numpy(None)
