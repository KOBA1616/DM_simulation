import time
import os
import gc
from types import ModuleType
from dm_toolkit.gui.i18n import tr

# Import Backend Modules
dm_ai_module: ModuleType | None
try:
    import dm_ai_module as _dm_ai_module  # type: ignore
    dm_ai_module = _dm_ai_module
except ImportError:
    dm_ai_module = None

from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.training.scenario_definitions import SCENARIOS

class SimulationRunner:
    def __init__(self, card_db, scenario_name, episodes, threads, sims, evaluator_type, model_path=None, deck_lists=None):
        self.card_db = card_db
        self.scenario_name = scenario_name
        self.episodes = episodes
        self.threads = threads
        self.sims = sims
        self.evaluator_type = evaluator_type # "Random", "Heuristic", "Model"
        self.model_path = model_path
        self.deck_lists = deck_lists
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

    def run(self, progress_callback=None, finished_callback=None):
        """
        Runs the simulation.
        progress_callback: Callable[[int, str], None]
        finished_callback: Callable[[float, str], None]
        """
        try:
            if not EngineCompat.is_available():
                if finished_callback:
                    finished_callback(0.0, tr("Error: dm_ai_module not loaded."))
                return

            assert dm_ai_module is not None

            if progress_callback:
                progress_callback(0, tr("Initializing..."))

            config = None
            if not self.deck_lists:
                # Setup Scenario
                if self.scenario_name not in SCENARIOS:
                    msg = f"Error: Unknown scenario {self.scenario_name}"
                    try:
                        msg = tr("Error: Unknown scenario {name}").format(name=self.scenario_name)
                    except Exception:
                        pass
                    if finished_callback:
                        finished_callback(0.0, msg)
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

            # Keep references to objects to prevent GC
            self.neural_evaluator = None
            self.torch_network = None

            if self.evaluator_type == "Model":
                try:
                    import torch
                    import numpy as np
                    from dm_toolkit.ai.agent.transformer_model import DuelTransformer

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    vocab_size = 1000
                    action_dim = 600
                    d_model = 256
                    nhead = 8
                    num_layers = 6
                    dim_feedforward = 1024
                    max_len = 200

                    self.torch_network = DuelTransformer(
                        vocab_size=vocab_size,
                        action_dim=action_dim,
                        d_model=d_model,
                        nhead=nhead,
                        num_layers=num_layers,
                        dim_feedforward=dim_feedforward,
                        max_len=max_len
                    ).to(device)

                    if self.model_path and os.path.exists(self.model_path):
                        checkpoint = torch.load(self.model_path, map_location=device)
                        # Handle both full checkpoint dict and direct state_dict
                        state_dict = checkpoint.get('model_state_dict', checkpoint)
                        self.torch_network.load_state_dict(state_dict)
                        msg = f"Loaded model from {self.model_path}"
                        try:
                            msg = tr("Loaded model from {path}").format(path=self.model_path)
                        except Exception:
                            pass
                        if progress_callback:
                            progress_callback(5, msg)
                    else:
                        if progress_callback:
                            progress_callback(5, tr("Using initialized model (Untrained)"))

                    self.torch_network.eval()

                    def model_batch_evaluate(states):
                        sequences = []
                        phases = []
                        legal_masks = []
                        reserved_dim = self.torch_network.reserved_dim

                        for s in states:
                            # Convert state to token sequence
                            seq = dm_ai_module.TensorConverter.convert_to_sequence(s, s.active_player_id, self.card_db)
                            sequences.append(seq)
                            phases.append(int(getattr(s, 'current_phase', 0)))

                            # Generate legal mask
                    try:
                        # Prefer command-first generator for legal mask; fallback via commands_v2 shim
                        from dm_toolkit import commands_v2 as commands
                        try:
                            legal_actions = commands.generate_legal_commands(s, self.card_db, strict=False) or []
                        except Exception:
                            try:
                                legal_actions = commands.generate_legal_commands(s, self.card_db) or []
                            except Exception:
                                legal_actions = []

                        mask = np.zeros(reserved_dim, dtype=bool)
                        from dm_toolkit.action_to_command import map_action
                        for action in (legal_actions or []):
                            try:
                                # Normalize to a command-dict if possible
                                if hasattr(action, 'to_dict'):
                                    d = action.to_dict()
                                elif isinstance(action, dict):
                                    d = action
                                else:
                                    try:
                                        d = map_action(action)
                                    except Exception:
                                        d = None

                                idx = dm_ai_module.CommandEncoder.command_to_index(d if d is not None else (action if isinstance(action, dict) else None))
                                if idx is not None and 0 <= idx < reserved_dim:
                                    mask[idx] = True
                            except Exception:
                                continue
                        legal_masks.append(mask)
                    except Exception:
                        # Fallback to permissive mask
                        legal_masks.append(np.ones(reserved_dim, dtype=bool))

                        # Pad sequences
                        # Assuming convert_to_sequence returns list of ints. Pad with 0.
                        max_len_batch = 200
                        padded_seqs = []
                        for seq in sequences:
                            if seq is None:
                                s_list = [0] * max_len_batch
                            else:
                                s_list = list(seq)
                                if len(s_list) > max_len_batch:
                                    s_list = s_list[:max_len_batch]
                                else:
                                    s_list = s_list + [0] * (max_len_batch - len(s_list))
                            padded_seqs.append(s_list)

                        input_tensor = torch.tensor(padded_seqs, dtype=torch.long).to(device)
                        padding_mask = (input_tensor == 0)
                        phase_ids = torch.tensor(phases, dtype=torch.long).to(device)
                        legal_mask_tensor = torch.tensor(np.array(legal_masks), dtype=torch.bool).to(device)

                        with torch.no_grad():
                            policy_logits, values = self.torch_network(input_tensor, padding_mask=padding_mask, phase_ids=phase_ids, legal_action_mask=legal_mask_tensor)
                            policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
                            vals = values.squeeze(1).cpu().numpy()

                        results = []
                        for i in range(len(states)):
                            results.append((policies[i], float(vals[i])))
                        return results

                    evaluator_func = model_batch_evaluate

                    # Register callback for Native C++ MCTS integration
                    EngineCompat.register_batch_inference_numpy(model_batch_evaluate)

                except ImportError:
                    if finished_callback:
                        finished_callback(0.0, tr("Error: PyTorch not available for Model evaluation."))
                    return
                except Exception as e:
                    msg = f"Error loading model: {e}"
                    try:
                        msg = tr("Error loading model: {e}").format(e=e)
                    except Exception:
                        pass
                    if finished_callback:
                        finished_callback(0.0, msg)
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

            if progress_callback:
                progress_callback(10, tr("Starting simulation") + "...")

            # Simulation parameters
            batch_size = 32

            # Chunking Strategy for Large Scale Simulations
            # Split total episodes into smaller chunks to manage memory
            chunk_size = 50 # Process 50 games at a time
            total_episodes = self.episodes
            num_chunks = (total_episodes + chunk_size - 1) // chunk_size

            all_results = []

            start_time = time.time()

            for chunk_idx in range(num_chunks):
                if self.is_cancelled:
                    if progress_callback:
                        progress_callback(int((chunk_idx / num_chunks) * 90), tr("Simulation cancelled."))
                    break

                # Determine chunk range
                start_game_idx = chunk_idx * chunk_size
                end_game_idx = min((chunk_idx + 1) * chunk_size, total_episodes)
                current_chunk_size = end_game_idx - start_game_idx

                if progress_callback:
                    msg = f"Processing chunk {chunk_idx + 1}/{num_chunks} ({current_chunk_size} games)..."
                    try:
                        msg = tr("Processing chunk {idx}/{num} ({count} games)...").format(idx=chunk_idx + 1, num=num_chunks, count=current_chunk_size)
                    except Exception:
                        pass
                    progress_callback(10 + int((chunk_idx / num_chunks) * 80), msg)

                # Prepare Initial States for this chunk
                chunk_initial_states = []
                for i in range(current_chunk_size):
                    global_idx = start_game_idx + i
                    seed = int(time.time() * 1000 + global_idx) % 1000000
                    state = dm_ai_module.GameState(seed)

                    if self.deck_lists:
                        state.setup_test_duel()
                        state.set_deck(0, self.deck_lists[0])
                        state.set_deck(1, self.deck_lists[1])
                        dm_ai_module.PhaseManager.start_game(state, self.card_db)
                    else:
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
                except Exception as e:
                    if finished_callback:
                        finished_callback(0.0, f"{tr('Simulation Error')} in chunk {chunk_idx}: {e}")
                    return
                finally:
                    # Clean up runner explicitly
                    del runner
                    gc.collect()

            if self.is_cancelled and not all_results:
                 if finished_callback:
                     finished_callback(0.0, tr("Simulation cancelled by user."))
                 return

            duration = time.time() - start_time

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

            summary = (
                f"{tr('Completed')} {total} episodes in {duration:.2f}s.\n"
                f"{tr('Wins')}: {wins} ({win_rate:.1f}%)\n"
                f"{tr('Losses')}: {losses}\n"
                f"{tr('Draws')}: {draws}\n"
                f"{tr('Throughput')}: {total/duration:.1f} games/s"
            )

            if finished_callback:
                finished_callback(win_rate, summary)

        finally:
            # Cleanup Callback
            if self.evaluator_type == "Model":
                # Unregister callback to prevent memory leaks and crash on exit
                EngineCompat.register_batch_inference_numpy(None)
