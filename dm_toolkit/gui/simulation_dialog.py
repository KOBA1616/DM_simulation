# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path
import time
import numpy as np
import gc
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox,
    QPushButton, QProgressBar, QTextEdit, QGroupBox, QMessageBox, QFileDialog,
    QCheckBox, QLineEdit, QDoubleSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
import threading
from dm_toolkit.gui.i18n import tr

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
import subprocess
import shlex
import json

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
        try:
            if not EngineCompat.is_available():
                self.finished_signal.emit(0.0, tr("Error: dm_ai_module not loaded."))
                return

            # Tell mypy that dm_ai_module is available after EngineCompat check
            assert dm_ai_module is not None

            self.progress_signal.emit(0, tr("Initializing..."))

            # Setup Scenario
            if self.scenario_name not in SCENARIOS:
                try:
                    self.finished_signal.emit(0.0, tr("Error: Unknown scenario {name}").format(name=self.scenario_name))
                except Exception:
                    self.finished_signal.emit(0.0, f"Error: Unknown scenario {self.scenario_name}")
                return
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
                        try:
                            self.progress_signal.emit(5, tr("Loaded model from {path}").format(path=self.model_path))
                        except Exception:
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
                    EngineCompat.register_batch_inference_numpy(batch_inference)

                    self.neural_evaluator = dm_ai_module.NeuralEvaluator(self.card_db)
                    evaluator_func = self.neural_evaluator.evaluate

                except ImportError:
                    self.finished_signal.emit(0.0, tr("Error: PyTorch not available for Model evaluation."))
                    return
                except Exception as e:
                    try:
                        self.finished_signal.emit(0.0, tr("Error loading model: {e}").format(e=e))
                    except Exception:
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
                    self.progress_signal.emit(int((chunk_idx / num_chunks) * 90), tr("Simulation cancelled."))
                    break

                # Determine chunk range
                start_game_idx = chunk_idx * chunk_size
                end_game_idx = min((chunk_idx + 1) * chunk_size, total_episodes)
                current_chunk_size = end_game_idx - start_game_idx

                try:
                    self.progress_signal.emit(
                        10 + int((chunk_idx / num_chunks) * 80),
                        tr("Processing chunk {idx}/{num} ({count} games)...").format(idx=chunk_idx + 1, num=num_chunks, count=current_chunk_size)
                    )
                except Exception:
                    self.progress_signal.emit(
                        10 + int((chunk_idx / num_chunks) * 80),
                        f"Processing chunk {chunk_idx + 1}/{num_chunks} ({current_chunk_size} games)..."
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
                # Recreating the runner helps to clear any internal buffers in C++ side
                runner = EngineCompat.create_parallel_runner(self.card_db, self.sims, batch_size)

                try:
                    results_info = EngineCompat.ParallelRunner_play_games(runner, chunk_initial_states, evaluator_func, 1.0, add_noise=False, threads=self.threads)
                    all_results.extend(results_info)
                except Exception as e:
                    self.finished_signal.emit(0.0, f"{tr('Simulation Error')} in chunk {chunk_idx}: {e}")
                    return
                finally:
                    # Clean up runner explicitly
                    del runner
                    gc.collect()

            if self.is_cancelled and not all_results:
                 self.finished_signal.emit(0.0, tr("Simulation cancelled by user."))
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

            self.finished_signal.emit(win_rate, summary)

        finally:
            # Cleanup Callback
            if self.evaluator_type == "Model":
                # Unregister callback to prevent memory leaks and crash on exit
                EngineCompat.register_batch_inference_numpy(None)


class AutoLoopWorker(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(float, str)

    def __init__(self, repo_root, iterations, episodes, epochs, batch_size, out_dir, keep_data, keep_models, parallel, win_threshold, run_for_seconds, eval_games=50, eval_parallel=8, eval_use_pytorch=False):
        super().__init__()
        self.repo_root = repo_root
        self.iterations = iterations
        self.episodes = episodes
        self.epochs = epochs
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.keep_data = keep_data
        self.keep_models = keep_models
        self.parallel = parallel
        self.win_threshold = win_threshold
        self.run_for_seconds = run_for_seconds
        # evaluation settings to run once after self-play completes
        self.eval_games = eval_games
        self.eval_parallel = eval_parallel
        self.eval_use_pytorch = eval_use_pytorch
        self.proc = None
        self._cancelled = False

    def run(self):
        cmd = [sys.executable, '-u', str(Path(self.repo_root) / 'training' / 'self_play_loop.py')]
        cmd += ['--iterations', str(self.iterations)]
        cmd += ['--episodes', str(self.episodes)]
        cmd += ['--epochs', str(self.epochs)]
        cmd += ['--batch-size', str(self.batch_size)]
        cmd += ['--out-dir', str(self.out_dir)]
        cmd += ['--keep-data', str(self.keep_data)]
        cmd += ['--keep-models', str(self.keep_models)]
        cmd += ['--parallel', str(self.parallel)]
        cmd += ['--win-threshold', str(self.win_threshold)]
        if self.run_for_seconds and self.run_for_seconds > 0:
            cmd += ['--run-for-seconds', str(self.run_for_seconds)]

        try:
            try:
                self.progress_signal.emit(0, tr("Running:") + ' ' + ' '.join(shlex.quote(x) for x in cmd))
            except Exception:
                self.progress_signal.emit(0, f"Running: {' '.join(shlex.quote(x) for x in cmd)}")

            self.proc = subprocess.Popen(cmd, cwd=self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            last_lines = []
            if self.proc.stdout:
                for line in self.proc.stdout:
                    if self._cancelled:
                        break
                    line = line.rstrip('\n')
                    last_lines.append(line)
                    if len(last_lines) > 200:
                        last_lines.pop(0)
                    # Emit as log (progress unknown)
                    self.progress_signal.emit(0, line)

            rc = None
            if not self._cancelled:
                rc = self.proc.wait()
            else:
                try:
                    self.proc.terminate()
                except Exception:
                    pass

            summary = '\n'.join(last_lines[-50:])
            # After self-play completes, run a single head2head evaluation using latest checkpoints
            try:
                if not self._cancelled:
                    h2h_script = Path(self.repo_root) / 'training' / 'head2head.py'
                    if h2h_script.exists():
                        # find checkpoints in repo_root/checkpoints/transformer
                        ck_dir = Path(self.repo_root) / 'checkpoints' / 'transformer'
                        models = []
                        if ck_dir.exists():
                            for fn in ck_dir.iterdir():
                                if fn.suffix in ['.pth', '.onnx'] and 'step_' in fn.name:
                                    models.append(fn)
                        # sort by step number parsed from filename
                        def step_key(p: Path):
                            parts = p.stem.split('_')
                            if 'step' in parts:
                                try:
                                    i = parts.index('step')
                                    return int(parts[i+1])
                                except Exception:
                                    return 0
                            return 0
                        models = sorted(models, key=step_key, reverse=True)
                        if len(models) >= 2:
                            model_a = str(models[0])
                            model_b = str(models[1])
                        elif len(models) == 1:
                            # only one model; skip evaluation
                            self.progress_signal.emit(0, tr('No baseline model found; skipping head2head eval.'))
                            model_a = None
                            model_b = None
                        else:
                            model_a = None
                            model_b = None

                        if model_a and model_b:
                            cmd = [sys.executable, str(h2h_script), model_a, model_b, '--games', str(self.eval_games), '--parallel', str(self.eval_parallel)]
                            if self.eval_use_pytorch:
                                cmd.append('--use_pytorch')
                            try:
                                try:
                                    self.progress_signal.emit(0, tr('Running head2head:') + ' ' + ' '.join(shlex.quote(x) for x in cmd))
                                except Exception:
                                    self.progress_signal.emit(0, f"Running head2head: {' '.join(shlex.quote(x) for x in cmd)}")
                                p = subprocess.Popen(cmd, cwd=self.repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                                h2h_lines = []
                                if p.stdout:
                                    for ln in p.stdout:
                                        if self._cancelled:
                                            break
                                        l = ln.rstrip('\n')
                                        h2h_lines.append(l)
                                        # forward to UI
                                        self.progress_signal.emit(0, l)
                                if not self._cancelled:
                                    try:
                                        p.wait()
                                    except Exception:
                                        pass
                                # append h2h summary to overall summary
                                summary = summary + '\n\n== head2head output ==\n' + '\n'.join(h2h_lines[-200:])
                            except Exception as e:
                                try:
                                    self.progress_signal.emit(0, tr('Failed to run head2head: {e}').format(e=e))
                                except Exception:
                                    self.progress_signal.emit(0, f'Failed to run head2head: {e}')
                    else:
                        self.progress_signal.emit(0, tr('head2head script not found; skipping eval.'))
            except Exception:
                pass

            self.finished_signal.emit(0.0, summary)
        except Exception as e:
            try:
                self.finished_signal.emit(0.0, tr('Auto-loop failed: {e}').format(e=e))
            except Exception:
                self.finished_signal.emit(0.0, f'Auto-loop failed: {e}')

    def cancel(self):
        self._cancelled = True
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


class SimulationDialog(QDialog):
    def __init__(self, card_db, parent=None):
        super().__init__(parent)
        self.card_db = card_db
        self.setWindowTitle(tr("Batch Simulation / Verification"))
        self.resize(600, 500)
        self.init_ui()
        try:
            self.load_settings()
        except Exception:
            pass
        self.worker = None

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Settings
        group = QGroupBox(tr("Settings"))
        form = QVBoxLayout(group)

        # Scenario
        h_scen = QHBoxLayout()
        lbl_scen = QLabel(tr("Scenario") + ":")
        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(list(SCENARIOS.keys()))
        self.scenario_combo.setToolTip(tr("Select the game scenario to simulate"))
        lbl_scen.setBuddy(self.scenario_combo)
        h_scen.addWidget(lbl_scen)
        h_scen.addWidget(self.scenario_combo, 1)
        form.addLayout(h_scen)

        # Evaluator
        h_eval = QHBoxLayout()
        lbl_eval = QLabel(tr("Evaluator") + ":")
        self.eval_combo = QComboBox()
        self.eval_combo.addItems([tr("Heuristic"), tr("Random"), tr("Model")])
        self.eval_combo.setToolTip(tr("Select the AI agent type for evaluation"))
        lbl_eval.setBuddy(self.eval_combo)
        h_eval.addWidget(lbl_eval)
        h_eval.addWidget(self.eval_combo, 1)
        form.addLayout(h_eval)

        # Parameters
        h_params = QHBoxLayout()

        lbl_episodes = QLabel(tr("Games") + ":")
        self.episodes_spin = QSpinBox()
        self.episodes_spin.setRange(1, 10000)
        self.episodes_spin.setValue(100)
        self.episodes_spin.setToolTip(tr("Total number of games to simulate"))
        lbl_episodes.setBuddy(self.episodes_spin)
        h_params.addWidget(lbl_episodes)
        h_params.addWidget(self.episodes_spin)

        lbl_threads = QLabel(tr("Threads") + ":")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 32)
        self.threads_spin.setValue(4)
        self.threads_spin.setToolTip(tr("Number of CPU threads to use"))
        lbl_threads.setBuddy(self.threads_spin)
        h_params.addWidget(lbl_threads)
        h_params.addWidget(self.threads_spin)

        lbl_sims = QLabel(tr("MCTS Sims") + ":")
        self.sims_spin = QSpinBox()
        self.sims_spin.setRange(10, 5000)
        self.sims_spin.setValue(800)
        self.sims_spin.setToolTip(tr("Monte Carlo Tree Search simulations per move"))
        lbl_sims.setBuddy(self.sims_spin)
        h_params.addWidget(lbl_sims)
        h_params.addWidget(self.sims_spin)
        # Recommended params button
        self.recommend_btn = QPushButton(tr("推奨パラメータ適用"))
        self.recommend_btn.setToolTip(tr("一括シミュレーションの推奨パラメータを設定します (軽量評価 / 長時間評価用の既定値)"))
        self.recommend_btn.clicked.connect(self.apply_recommended_parameters)
        h_params.addWidget(self.recommend_btn)

        form.addLayout(h_params)
        # Auto-loop settings
        loop_group = QGroupBox(tr("Auto-Loop (Train & Eval)"))
        loop_layout = QHBoxLayout(loop_group)

        self.auto_checkbox = QCheckBox(tr("Enable Auto-Loop"))
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 10000)
        self.iterations_spin.setValue(3)
        self.iterations_spin.setToolTip(tr("Number of iterations for the auto loop"))

        # run duration in minutes
        self.run_minutes_spin = QSpinBox()
        self.run_minutes_spin.setRange(0, 24 * 60)
        self.run_minutes_spin.setValue(0)
        self.run_minutes_spin.setToolTip(tr("停止するまでの実行時間（分単位、0=無効）"))

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(1)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(8)

        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 128)
        self.parallel_spin.setValue(8)
        self.parallel_spin.setToolTip(tr("Number of parallel games per head-to-head batch"))

        self.keep_data_spin = QSpinBox()
        self.keep_data_spin.setRange(0, 1000)
        self.keep_data_spin.setValue(5)

        self.keep_models_spin = QSpinBox()
        self.keep_models_spin.setRange(0, 1000)
        self.keep_models_spin.setValue(3)

        self.win_threshold_spin = QDoubleSpinBox()
        self.win_threshold_spin.setRange(0.0, 1.0)
        self.win_threshold_spin.setSingleStep(0.01)
        self.win_threshold_spin.setValue(0.55)

        loop_layout.addWidget(self.auto_checkbox)
        loop_layout.addWidget(QLabel(tr("Iterations")))
        loop_layout.addWidget(self.iterations_spin)
        loop_layout.addWidget(QLabel(tr("Run minutes")))
        loop_layout.addWidget(self.run_minutes_spin)
        loop_layout.addWidget(QLabel(tr("Epochs")))
        loop_layout.addWidget(self.epochs_spin)
        loop_layout.addWidget(QLabel(tr("Batch")))
        loop_layout.addWidget(self.batch_spin)
        loop_layout.addWidget(QLabel(tr("Parallel")))
        loop_layout.addWidget(self.parallel_spin)
        loop_layout.addWidget(QLabel(tr("Keep data")))
        loop_layout.addWidget(self.keep_data_spin)
        loop_layout.addWidget(QLabel(tr("Keep models")))
        loop_layout.addWidget(self.keep_models_spin)
        loop_layout.addWidget(QLabel(tr("Win thresh")))
        loop_layout.addWidget(self.win_threshold_spin)

        form.addWidget(loop_group)
        layout.addWidget(group)

        # Head2Head evaluation quick-run group
        h2h_group = QGroupBox(tr("head2head 評価 (手動実行)"))
        h2h_layout = QHBoxLayout(h2h_group)

        self.h2h_games_spin = QSpinBox()
        self.h2h_games_spin.setRange(1, 5000)
        self.h2h_games_spin.setValue(50)
        self.h2h_games_spin.setToolTip(tr("評価で実行するゲーム数 (軽量評価は 20-100 推奨)"))

        self.h2h_parallel_spin = QSpinBox()
        self.h2h_parallel_spin.setRange(1, 256)
        self.h2h_parallel_spin.setValue(8)
        self.h2h_parallel_spin.setToolTip(tr("head2head の parallel 値 (バッチサイズ)"))

        self.h2h_use_pytorch_cb = QCheckBox(tr("PyTorch を使用"))
        self.h2h_use_pytorch_cb.setToolTip(tr("PyTorch チェックポイント (.pth) を読み込んで評価する場合はチェック"))

        self.h2h_baseline_le = QLineEdit()
        self.h2h_baseline_le.setToolTip(tr("ベースラインモデルのパス (onnx/.pth)。空の場合は未指定。"))
        self.h2h_baseline_btn = QPushButton(tr("参照"))
        def pick_baseline():
            f, _ = QFileDialog.getOpenFileName(self, tr("ベースラインモデルを選択"), os.getcwd(), "Model Files (*.onnx *.pth);;All Files (*)")
            if f:
                self.h2h_baseline_le.setText(f)
        self.h2h_baseline_btn.clicked.connect(pick_baseline)

        self.h2h_run_btn = QPushButton(tr("Run head2head 今すぐ実行"))
        self.h2h_run_btn.setToolTip(tr("現在の設定で head2head を実行し、ログに出力します。"))
        self.h2h_run_btn.clicked.connect(self.run_head2head_now)

        h2h_layout.addWidget(QLabel(tr("ゲーム数:")))
        h2h_layout.addWidget(self.h2h_games_spin)
        h2h_layout.addWidget(QLabel(tr("parallel:")))
        h2h_layout.addWidget(self.h2h_parallel_spin)
        h2h_layout.addWidget(self.h2h_use_pytorch_cb)
        h2h_layout.addWidget(self.h2h_baseline_le)
        h2h_layout.addWidget(self.h2h_baseline_btn)
        h2h_layout.addWidget(self.h2h_run_btn)

        layout.addWidget(h2h_group)
        # Auto-eval settings editable
        eval_group = QGroupBox(tr("head2head 自動評価設定"))
        eval_layout = QHBoxLayout(eval_group)
        self.eval_every_steps_spin = QSpinBox()
        self.eval_every_steps_spin.setRange(0, 100000000)
        self.eval_every_steps_spin.setValue(0)
        self.eval_every_steps_spin.setToolTip(tr("0=無効。チェックポイント保存後にこのステップ毎に評価を実行します（ステップ単位）。"))
        eval_layout.addWidget(QLabel(tr("評価頻度 (steps):")))
        eval_layout.addWidget(self.eval_every_steps_spin)
        eval_layout.addWidget(QLabel(tr("評価ゲーム数:")))
        self.eval_games_spin = QSpinBox()
        self.eval_games_spin.setRange(1, 5000)
        self.eval_games_spin.setValue(50)
        eval_layout.addWidget(self.eval_games_spin)
        eval_layout.addWidget(QLabel(tr("parallel:")))
        self.eval_parallel_spin = QSpinBox()
        self.eval_parallel_spin.setRange(1, 256)
        self.eval_parallel_spin.setValue(8)
        eval_layout.addWidget(self.eval_parallel_spin)
        self.eval_use_pytorch_cb2 = QCheckBox(tr("PyTorch を使用"))
        eval_layout.addWidget(self.eval_use_pytorch_cb2)
        eval_layout.addStretch()
        layout.addWidget(eval_group)

        # Warning label for memory leak
        leak_warning = QLabel(tr("Note: High simulation counts may cause memory issues (std::bad_alloc)."))
        leak_warning.setStyleSheet("color: orange; font-style: italic;")
        layout.addWidget(leak_warning)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.run_btn = QPushButton(tr("Run Simulation"))
        self.run_btn.setToolTip(tr("Start the batch simulation with current settings"))
        self.run_btn.clicked.connect(self.start_simulation)

        self.cancel_btn = QPushButton(tr("Cancel"))
        self.cancel_btn.setToolTip(tr("Close this dialog"))
        self.cancel_btn.clicked.connect(self.cancel_simulation)
        self.cancel_btn.setEnabled(True) # Always enabled to close dialog

        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        # Progress and Log
        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        # Status panel (visual summary)
        status_group = QGroupBox(tr("Simulation Status"))
        status_layout = QHBoxLayout(status_group)
        self.status_label = QLabel(tr("Ready"))
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
        status_layout.addWidget(self.status_label)

        self.wins_label = QLabel(tr("Wins") + ": 0")
        status_layout.addWidget(self.wins_label)

        self.losses_label = QLabel(tr("Losses") + ": 0")
        status_layout.addWidget(self.losses_label)

        self.draws_label = QLabel(tr("Draws") + ": 0")
        status_layout.addWidget(self.draws_label)

        self.throughput_label = QLabel(tr("Throughput") + ": 0.0 games/s")
        status_layout.addWidget(self.throughput_label)

        layout.addWidget(status_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        # Time progress / remaining display for run_minutes
        time_layout = QHBoxLayout()
        self.remaining_label = QLabel(tr("Remaining") + ": -")
        self.time_progress = QProgressBar()
        self.time_progress.setRange(0, 100)
        self.time_progress.setValue(0)
        time_layout.addWidget(self.remaining_label)
        time_layout.addWidget(self.time_progress)
        layout.addLayout(time_layout)

    def start_simulation(self):
        scenario = self.scenario_combo.currentText()
        evaluator = self.eval_combo.currentText()
        episodes = self.episodes_spin.value()
        threads = self.threads_spin.value()
        sims = self.sims_spin.value()
        try:
            self.save_settings()
        except Exception:
            pass

        model_path = None
        if evaluator == "Model":
            file_path, _ = QFileDialog.getOpenFileName(self, tr("Select Model File"), "", "Model Files (*.pth);;All Files (*)")
            if file_path:
                model_path = file_path
            else:
                return

        try:
            self.log_text.append(tr("Starting simulation: {scenario}, {evaluator}, {episodes} games...").format(scenario=scenario, evaluator=evaluator, episodes=episodes))
        except Exception:
            self.log_text.append(f"{tr('Starting simulation')}: {scenario}, {evaluator}, {episodes} games...")
        self.run_btn.setEnabled(False)
        self.progress.setValue(0)
        # If auto-loop enabled, run the training loop script as subprocess
        if getattr(self, 'auto_checkbox', None) and self.auto_checkbox.isChecked():
            repo_root = str(Path(__file__).resolve().parents[2])
            iterations = self.iterations_spin.value()
            run_seconds = int(self.run_minutes_spin.value()) * 60
            epochs = self.epochs_spin.value()
            batch_size = self.batch_spin.value()
            out_dir = os.path.join(repo_root, 'data')
            keep_data = self.keep_data_spin.value()
            keep_models = self.keep_models_spin.value()
            win_threshold = self.win_threshold_spin.value()

            parallel = self.parallel_spin.value()
            # pass eval settings so AutoLoopWorker can run one evaluation after self-play
            self.auto_worker = AutoLoopWorker(
                repo_root, iterations, episodes, epochs, batch_size, out_dir, keep_data, keep_models, parallel, win_threshold, run_seconds,
                eval_games=self.eval_games_spin.value(), eval_parallel=self.eval_parallel_spin.value(), eval_use_pytorch=self.eval_use_pytorch_cb2.isChecked()
            )
            self.auto_worker.progress_signal.connect(self.update_progress)
            self.auto_worker.finished_signal.connect(self.simulation_finished)
            self.auto_worker.start()
            # track start time for remaining-time UI
            try:
                self.auto_start_time = time.time()
                self.auto_run_seconds = run_seconds
                # start a timer to update remaining time every second
                try:
                    if getattr(self, '_time_timer', None) is None:
                        self._time_timer = QTimer(self)
                        self._time_timer.timeout.connect(self._update_time_remaining)
                    self._time_timer.start(1000)
                except Exception:
                    pass
            except Exception:
                self.auto_start_time = None
                self.auto_run_seconds = 0
            return

        self.worker = SimulationWorker(self.card_db, scenario, episodes, threads, sims, evaluator, model_path)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.simulation_finished)
        self.worker.start()

    def run_head2head_now(self):
        # Build head2head command and run as subprocess, forwarding lines to log_text
        repo_root = str(Path(__file__).resolve().parents[2])
        h2h_script = os.path.join(repo_root, 'training', 'head2head.py')
        if not os.path.exists(h2h_script):
            self.log_text.append(tr("Error: head2head script not found: {p}").format(p=h2h_script))
            return

        # Choose model paths: prompt for two models (P0 vs baseline)
        # For convenience, first model can be selected via file dialog
        model_a, _ = QFileDialog.getOpenFileName(self, tr("評価モデルを選択 (Player A)"), os.getcwd(), "Model Files (*.pth *.onnx);;All Files (*)")
        if not model_a:
            return
        model_b = self.h2h_baseline_le.text().strip()
        if not model_b:
            # attempt auto-select strongest model from checkpoints/transformer
            repo_root = str(Path(__file__).resolve().parents[2])
            ck_dir = os.path.join(repo_root, 'checkpoints', 'transformer')
            best = None
            best_step = -1
            try:
                if os.path.exists(ck_dir):
                    for fn in os.listdir(ck_dir):
                        if fn.endswith('.pth') and 'step_' in fn:
                            parts = fn.split('_')
                            try:
                                idx = parts.index('step')
                                s = int(parts[idx+1].split('.')[0])
                                if s > best_step:
                                    best_step = s
                                    best = os.path.join(ck_dir, fn)
                            except Exception:
                                continue
            except Exception:
                best = None
            if best:
                model_b = best
                self.h2h_baseline_le.setText(model_b)
            else:
                # prompt to choose baseline if empty
                model_b, _ = QFileDialog.getOpenFileName(self, tr("ベースラインモデルを選択 (Player B)"), os.getcwd(), "Model Files (*.pth *.onnx);;All Files (*)")
                if not model_b:
                    return

        games = str(self.h2h_games_spin.value())
        parallel = str(self.h2h_parallel_spin.value())
        use_pytorch = self.h2h_use_pytorch_cb.isChecked()

        cmd = [sys.executable, h2h_script, model_a, model_b, '--games', games, '--parallel', parallel]
        if use_pytorch:
            cmd.append('--use_pytorch')

        self.log_text.append(tr("Running head2head: {cmd}").format(cmd=' '.join(cmd)))

        try:
            proc = subprocess.Popen(cmd, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except Exception as e:
            self.log_text.append(f"Failed to start head2head: {e}")
            return

        def reader_thread():
            try:
                if proc.stdout:
                    for ln in proc.stdout:
                        line = ln.rstrip('\n')
                        # emit to GUI thread
                        QTimer.singleShot(0, lambda l=line: self.log_text.append(l))
            except Exception as e:
                QTimer.singleShot(0, lambda: self.log_text.append(f"head2head read error: {e}"))

        t = threading.Thread(target=reader_thread, daemon=True)
        t.start()

    def cancel_simulation(self):
        if getattr(self, 'auto_worker', None):
            try:
                self.auto_worker.cancel()
            except Exception:
                pass
            self.log_text.append(tr("Cancelling auto-loop..."))
        if getattr(self, 'worker', None):
            self.worker.is_cancelled = True
            self.log_text.append(tr("Cancelling..."))
        self.reject()
        try:
            if getattr(self, '_time_timer', None):
                self._time_timer.stop()
        except Exception:
            pass
        try:
            self.auto_start_time = None
            self.auto_run_seconds = 0
        except Exception:
            pass

    def update_progress(self, val, msg):
        # Defensive: coerce types so UI updates don't fail due to numpy/int/bytes
        try:
            try:
                ival = int(val)
            except Exception:
                ival = None
            if ival is not None:
                try:
                    # clamp to 0-100
                    ival = max(0, min(100, int(ival)))
                    self.progress.setValue(ival)
                except Exception:
                    pass
        except Exception:
            pass

        # Append raw log for detail (always convert to str)
        try:
            msg_str = str(msg)
        except Exception:
            msg_str = ''
        try:
            self.log_text.append(msg_str)
        except Exception:
            pass

        # If a JSON structured progress line is emitted (from head2head), parse it.
        # Format: H2H_JSON: { ... }
        try:
            if msg_str.startswith('H2H_JSON:'):
                jtxt = msg_str.split(':', 1)[1].strip()
                data = json.loads(jtxt)
                ev = data.get('event')
                if ev == 'progress':
                    wins = data.get('wins', 0)
                    losses = data.get('losses', 0)
                    draws = data.get('draws', 0)
                    try:
                        self.wins_label.setText(tr('Wins') + f": {int(wins)}")
                        self.losses_label.setText(tr('Losses') + f": {int(losses)}")
                        self.draws_label.setText(tr('Draws') + f": {int(draws)}")
                        self.status_label.setText(tr('Processing'))
                        self.status_label.setStyleSheet("font-weight: bold; color: orange;")
                    except Exception:
                        pass
                elif ev == 'summary':
                    wins = data.get('wins', 0)
                    losses = data.get('losses', 0)
                    draws = data.get('draws', 0)
                    try:
                        self.wins_label.setText(tr('Wins') + f": {int(wins)}")
                        self.losses_label.setText(tr('Losses') + f": {int(losses)}")
                        self.draws_label.setText(tr('Draws') + f": {int(draws)}")
                        self.status_label.setText(tr('Completed'))
                        self.status_label.setStyleSheet("font-weight: bold; color: green;")
                    except Exception:
                        pass
                elif ev == 'h2h_stats':
                    try:
                        avg_len = data.get('avg_game_length')
                        pass_rate = data.get('pass_rate')
                        try:
                            if avg_len is not None:
                                self.throughput_label.setText(tr('AvgLen') + f": {float(avg_len):.2f} moves")
                            else:
                                self.throughput_label.setText(tr('AvgLen') + ': -')
                        except Exception:
                            pass
                        try:
                            if pass_rate is not None:
                                self.log_text.append(tr('PASS率') + f": {float(pass_rate):.3f}")
                        except Exception:
                            pass
                        # optionally pretty-print turn_action_distribution
                        tad = data.get('turn_action_distribution')
                        if tad:
                            try:
                                self.log_text.append(tr('Turn action distribution:'))
                                self.log_text.append(json.dumps(tad, ensure_ascii=False))
                            except Exception:
                                pass
                    except Exception:
                        pass
                # we've handled the JSON message; return early
                return
        except Exception:
            pass

        # Lightweight parsing to update visual status
        try:
            lower = msg.lower()
        except Exception:
            lower = msg

        # Chunk processing -> update status
        if "processing chunk" in lower or "チャンク処理" in msg:
            try:
                self.status_label.setText(tr("Processing"))
                self.status_label.setStyleSheet("font-weight: bold; color: orange;")
            except Exception:
                self.status_label.setText("Processing")
                self.status_label.setStyleSheet("font-weight: bold; color: orange;")

        # Update remaining time UI if auto-loop with time limit is running
        try:
            if getattr(self, 'auto_start_time', None) and getattr(self, 'auto_run_seconds', 0) and self.auto_run_seconds > 0:
                self._update_time_remaining()
        except Exception:
            pass

        # Running command line
        if lower.startswith("running:") or lower.startswith(tr("Running:").lower()):
            self.status_label.setText(tr("Running:"))
            self.status_label.setStyleSheet("font-weight: bold; color: blue;")

        # If message contains simple stats (single-line), try to extract numbers
        # Wins / Losses / Draws patterns
        import re
        m_w = re.search(r"wins?:\s*(\d+)", lower)
        m_l = re.search(r"losses?:\s*(\d+)", lower)
        m_d = re.search(r"draws?:\s*(\d+)", lower)
        if m_w:
            try:
                self.wins_label.setText(tr("Wins") + f": {int(m_w.group(1))}")
            except Exception:
                pass
        if m_l:
            try:
                self.losses_label.setText(tr("Losses") + f": {int(m_l.group(1))}")
            except Exception:
                pass
        if m_d:
            try:
                self.draws_label.setText(tr("Draws") + f": {int(m_d.group(1))}")
            except Exception:
                pass

    def simulation_finished(self, win_rate, summary):
        self.progress.setValue(100)
        self.log_text.append("=== " + tr("Completed") + " ===")
        self.log_text.append(summary)

        # Parse summary block for Wins / Losses / Draws / Throughput
        try:
            import re
            wins = re.search(r"Wins\D*(\d+)", summary)
            losses = re.search(r"Losses\D*(\d+)", summary)
            draws = re.search(r"Draws\D*(\d+)", summary)
            thr = re.search(r"Throughput\D*([0-9\.]+)", summary)
            if wins:
                self.wins_label.setText(tr("Wins") + f": {int(wins.group(1))}")
            if losses:
                self.losses_label.setText(tr("Losses") + f": {int(losses.group(1))}")
            if draws:
                self.draws_label.setText(tr("Draws") + f": {int(draws.group(1))}")
            if thr:
                self.throughput_label.setText(tr("Throughput") + f": {float(thr.group(1)):.1f} games/s")
        except Exception:
            pass

        self.run_btn.setEnabled(True)
        self.worker = None
        try:
            if getattr(self, '_time_timer', None):
                self._time_timer.stop()
        except Exception:
            pass
        try:
            self.auto_start_time = None
            self.auto_run_seconds = 0
            self.time_progress.setValue(0)
            self.remaining_label.setText(tr("Remaining") + ": -")
        except Exception:
            pass

    # Persistence helpers for remembering last-used settings
    def settings_file_path(self) -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        return repo_root / 'data' / 'sim_settings.json'

    def _update_time_remaining(self):
        try:
            if not getattr(self, 'auto_start_time', None) or not getattr(self, 'auto_run_seconds', 0):
                return
            total = float(self.auto_run_seconds)
            elapsed = time.time() - float(self.auto_start_time)
            remaining = max(0.0, total - elapsed)
            # update label as MM:SS
            mins = int(remaining) // 60
            secs = int(remaining) % 60
            self.remaining_label.setText(tr('Remaining') + f": {mins:02d}:{secs:02d}")
            # update time progress as percent elapsed
            pct = int(min(100.0, (elapsed / total) * 100.0)) if total > 0 else 0
            self.time_progress.setValue(pct)
            if remaining <= 0:
                try:
                    self.status_label.setText(tr('Time up'))
                    self.status_label.setStyleSheet("font-weight: bold; color: red;")
                except Exception:
                    pass
        except Exception:
            pass

    def apply_recommended_parameters(self):
        """Apply recommended parameters for batch simulation and head2head.

        These defaults are tuned for a balance between speed and signal:
        - Games: 100
        - Threads: min(4, CPU)
        - MCTS sims: 800
        - head2head games: 50
        - head2head parallel: 8
        - eval games: 50
        - eval parallel: 8
        """
        try:
            import multiprocessing
            cpu = multiprocessing.cpu_count()
        except Exception:
            cpu = 4
        try:
            self.episodes_spin.setValue(100)
            self.threads_spin.setValue(min(4, cpu))
            self.sims_spin.setValue(800)
            self.h2h_games_spin.setValue(50)
            self.h2h_parallel_spin.setValue(8)
            self.h2h_use_pytorch_cb.setChecked(False)
            self.h2h_baseline_le.setText('')
            self.eval_every_steps_spin.setValue(5000)
            self.eval_games_spin.setValue(50)
            self.eval_parallel_spin.setValue(8)
            self.eval_use_pytorch_cb2.setChecked(False)
            # reasonable auto-loop defaults
            self.iterations_spin.setValue(3)
            self.batch_spin.setValue(8)
            self.parallel_spin.setValue(8)
            self.keep_data_spin.setValue(5)
            self.keep_models_spin.setValue(3)
            self.win_threshold_spin.setValue(0.55)
        except Exception:
            pass

    def load_settings(self):
        p = self.settings_file_path()
        if not p.exists():
            return
        with open(p, 'r', encoding='utf-8') as f:
            s = json.load(f)
        # Apply saved values defensively
        try:
            scen = s.get('scenario')
            if scen:
                for i in range(self.scenario_combo.count()):
                    if self.scenario_combo.itemText(i) == scen:
                        self.scenario_combo.setCurrentIndex(i)
                        break
        except Exception:
            pass
        try:
            ev = s.get('evaluator')
            if ev:
                for i in range(self.eval_combo.count()):
                    if self.eval_combo.itemText(i) == ev:
                        self.eval_combo.setCurrentIndex(i)
                        break
        except Exception:
            pass
        # Numeric fields
        try: self.episodes_spin.setValue(int(s.get('episodes', self.episodes_spin.value())))
        except Exception: pass
        try: self.threads_spin.setValue(int(s.get('threads', self.threads_spin.value())))
        except Exception: pass
        try: self.sims_spin.setValue(int(s.get('sims', self.sims_spin.value())))
        except Exception: pass
        try: self.auto_checkbox.setChecked(bool(s.get('auto_enabled', False)))
        except Exception: pass
        try: self.iterations_spin.setValue(int(s.get('iterations', self.iterations_spin.value())))
        except Exception: pass
        # legacy 'run_seconds' removed; use 'run_minutes' only
        try: self.epochs_spin.setValue(int(s.get('epochs', self.epochs_spin.value())))
        except Exception: pass
        try: self.batch_spin.setValue(int(s.get('batch', self.batch_spin.value())))
        except Exception: pass
        try: self.parallel_spin.setValue(int(s.get('parallel', self.parallel_spin.value())))
        except Exception: pass
        try: self.keep_data_spin.setValue(int(s.get('keep_data', self.keep_data_spin.value())))
        except Exception: pass
        try: self.keep_models_spin.setValue(int(s.get('keep_models', self.keep_models_spin.value())))
        except Exception: pass
        try: self.win_threshold_spin.setValue(float(s.get('win_threshold', self.win_threshold_spin.value())))
        except Exception: pass
        try: self.run_minutes_spin.setValue(int(s.get('run_minutes', self.run_minutes_spin.value())))
        except Exception: pass
        # head2head quick-run settings
        try: self.h2h_games_spin.setValue(int(s.get('h2h_games', self.h2h_games_spin.value())))
        except Exception: pass
        try: self.h2h_parallel_spin.setValue(int(s.get('h2h_parallel', self.h2h_parallel_spin.value())))
        except Exception: pass
        try: self.h2h_use_pytorch_cb.setChecked(bool(s.get('h2h_use_pytorch', self.h2h_use_pytorch_cb.isChecked())))
        except Exception: pass
        try: self.h2h_baseline_le.setText(str(s.get('h2h_baseline', self.h2h_baseline_le.text())))
        except Exception: pass
        try: self.eval_every_steps_spin.setValue(int(s.get('eval_every_steps', self.eval_every_steps_spin.value())))
        except Exception: pass
        try: self.eval_games_spin.setValue(int(s.get('eval_games', self.eval_games_spin.value())))
        except Exception: pass
        try: self.eval_parallel_spin.setValue(int(s.get('eval_parallel', self.eval_parallel_spin.value())))
        except Exception: pass
        try: self.eval_use_pytorch_cb2.setChecked(bool(s.get('eval_use_pytorch', self.eval_use_pytorch_cb2.isChecked())))
        except Exception: pass

    def save_settings(self):
        p = self.settings_file_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        s = {
            'scenario': self.scenario_combo.currentText(),
            'evaluator': self.eval_combo.currentText(),
            'episodes': self.episodes_spin.value(),
            'threads': self.threads_spin.value(),
            'sims': self.sims_spin.value(),
            'auto_enabled': bool(self.auto_checkbox.isChecked()),
            'iterations': self.iterations_spin.value(),
            'epochs': self.epochs_spin.value(),
            'batch': self.batch_spin.value(),
            'parallel': self.parallel_spin.value(),
            'keep_data': self.keep_data_spin.value(),
            'keep_models': self.keep_models_spin.value(),
            'win_threshold': self.win_threshold_spin.value(),
            # run duration in minutes
            'run_minutes': self.run_minutes_spin.value(),
            # head2head quick-run settings
            'h2h_games': self.h2h_games_spin.value(),
            'h2h_parallel': self.h2h_parallel_spin.value(),
            'h2h_use_pytorch': bool(self.h2h_use_pytorch_cb.isChecked()),
            'h2h_baseline': str(self.h2h_baseline_le.text()),
            # auto-eval settings
            'eval_every_steps': self.eval_every_steps_spin.value(),
            'eval_games': self.eval_games_spin.value(),
            'eval_parallel': self.eval_parallel_spin.value(),
            'eval_use_pytorch': bool(self.eval_use_pytorch_cb2.isChecked()),
        }
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
