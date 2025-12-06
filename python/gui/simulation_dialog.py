
import sys
import os
import time
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QProgressBar, QTextEdit, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from gui.localization import tr

# Import Backend Modules
try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

from training.scenario_definitions import SCENARIOS

# Worker Thread for Running Simulations
class SimulationWorker(QThread):
    progress_signal = pyqtSignal(int, str) # progress %, log message
    finished_signal = pyqtSignal(float, str) # win_rate, summary

    def __init__(self, card_db, scenario_name, episodes, threads, sims, evaluator_type, model_path=None):
        super().__init__()
        self.card_db = card_db
        self.scenario_name = scenario_name
        self.episodes = episodes
        self.threads = threads
        self.sims = sims
        self.evaluator_type = evaluator_type # "Random", "Heuristic", "Model"
        self.model_path = model_path
        self.is_cancelled = False

    def run(self):
        if not dm_ai_module:
            self.finished_signal.emit(0.0, tr("Error: dm_ai_module not loaded."))
            return

        self.progress_signal.emit(0, tr("Initializing..."))

        # Setup Scenario
        if self.scenario_name not in SCENARIOS:
            self.finished_signal.emit(0.0, f"Error: Unknown scenario {self.scenario_name}")
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

        # Prepare Initial States
        initial_states = []
        for i in range(self.episodes):
             instance = dm_ai_module.GameInstance(int(time.time() * 1000 + i) % 1000000, self.card_db)
             instance.reset_with_scenario(config)
             initial_states.append(instance.state)

        # Setup Evaluator
        evaluator_func = None

        # Keep references to objects to prevent GC
        self.neural_evaluator = None
        self.torch_network = None

        if self.evaluator_type == "Model":
            try:
                import torch
                from py_ai.agent.network import AlphaZeroNetwork

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Input Size check
                dummy_instance = dm_ai_module.GameInstance(42, self.card_db)
                dummy_vec = dm_ai_module.TensorConverter.convert_to_tensor(dummy_instance.state, 0, self.card_db)
                input_size = len(dummy_vec)
                action_size = 600

                self.torch_network = AlphaZeroNetwork(input_size, action_size).to(device)

                if self.model_path and os.path.exists(self.model_path):
                    self.torch_network.load_state_dict(torch.load(self.model_path, map_location=device))
                    self.progress_signal.emit(5, f"Loaded model from {self.model_path}")
                else:
                    self.progress_signal.emit(5, tr("Using initialized model (Untrained)"))

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
                dm_ai_module.register_batch_inference_numpy(batch_inference)

                self.neural_evaluator = dm_ai_module.NeuralEvaluator(self.card_db)
                evaluator_func = self.neural_evaluator.evaluate

            except ImportError:
                self.finished_signal.emit(0.0, "Error: PyTorch not available for Model evaluation.")
                return
            except Exception as e:
                self.finished_signal.emit(0.0, f"Error loading model: {e}")
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

        self.progress_signal.emit(10, tr("Starting simulation") + "...")

        runner = dm_ai_module.ParallelRunner(self.card_db, self.sims, 32) # Fixed batch size 32

        # Run
        start_time = time.time()

        try:
            results_info = runner.play_games(initial_states, evaluator_func, 1.0, False, self.threads)
        except Exception as e:
            self.finished_signal.emit(0.0, f"{tr('Simulation Error')}: {e}")
            return

        duration = time.time() - start_time

        # Tally results
        wins = 0
        losses = 0
        draws = 0

        for info in results_info:
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

        self.finished_signal.emit(win_rate, summary)

        # Cleanup
        if self.evaluator_type == "Model":
            if hasattr(dm_ai_module, "clear_batch_inference_numpy"):
                dm_ai_module.clear_batch_inference_numpy()

class SimulationDialog(QDialog):
    def __init__(self, card_db, parent=None):
        super().__init__(parent)
        self.card_db = card_db
        self.setWindowTitle(tr("Batch Simulation / Verification"))
        self.resize(600, 500)
        self.init_ui()
        self.worker = None

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Settings
        group = QGroupBox(tr("Settings"))
        form = QVBoxLayout(group)

        # Scenario
        h_scen = QHBoxLayout()
        h_scen.addWidget(QLabel(tr("Scenario") + ":"))
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(list(SCENARIOS.keys()))
        h_scen.addWidget(self.scenario_combo, 1)
        form.addLayout(h_scen)

        # Evaluator
        h_eval = QHBoxLayout()
        h_eval.addWidget(QLabel(tr("Evaluator") + ":"))
        self.eval_combo = QComboBox()
        self.eval_combo.addItems(["Heuristic", "Random", "Model"])
        h_eval.addWidget(self.eval_combo, 1)
        form.addLayout(h_eval)

        # Parameters
        h_params = QHBoxLayout()
        h_params.addWidget(QLabel(tr("Games") + ":"))
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 10000)
        self.episodes_spin.setValue(100)
        h_params.addWidget(self.episodes_spin)

        h_params.addWidget(QLabel(tr("Threads") + ":"))
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 32)
        self.threads_spin.setValue(4)
        h_params.addWidget(self.threads_spin)

        h_params.addWidget(QLabel(tr("MCTS Sims") + ":"))
        self.sims_spin = QSpinBox()
        self.sims_spin.setRange(10, 5000)
        self.sims_spin.setValue(800)
        h_params.addWidget(self.sims_spin)

        form.addLayout(h_params)
        layout.addWidget(group)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton(tr("Run Simulation"))
        self.run_btn.clicked.connect(self.start_simulation)
        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setEnabled(True) # Always enabled to close dialog

        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        # Progress and Log
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

    def start_simulation(self):
        scenario = self.scenario_combo.currentText()
        evaluator = self.eval_combo.currentText()
        episodes = self.episodes_spin.value()
        threads = self.threads_spin.value()
        sims = self.sims_spin.value()

        self.log_text.append(f"{tr('Starting simulation')}: {scenario}, {evaluator}, {episodes} games...")
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)

        # Todo: Allow model selection if Model is chosen
        model_path = None
        if evaluator == "Model":
            # For MVP, assume model_v1.pth or check default
            # Or ask user?
            # Let's check for model_v1.pth in current dir or project root
            # Or just let it fail/warn if not found
            potential_paths = ["model_v1.pth", "../model_v1.pth", "python/training/model_v1.pth"]
            for p in potential_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            if not model_path:
                self.log_text.append(tr("Warning: No model_v1.pth found. Using random weights."))

        self.worker = SimulationWorker(self.card_db, scenario, episodes, threads, sims, evaluator, model_path)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.simulation_finished)
        self.worker.start()

    def update_progress(self, val, msg):
        self.progress.setValue(val)
        self.log_text.append(msg)

    def simulation_finished(self, win_rate, summary):
        self.progress.setValue(100)
        self.log_text.append("=== " + tr("Completed") + " ===")
        self.log_text.append(summary)
        self.run_btn.setEnabled(True)
        self.worker = None
