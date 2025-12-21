# -*- coding: utf-8 -*-
import os
import sys
import json
import gc
import time
import random
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QTextEdit, QProgressBar,
    QComboBox, QCheckBox, QTabWidget, QWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush

# Adjust path to find dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../bin'))
try:
    import dm_ai_module
except ImportError:
    dm_ai_module = None

from gui.localization import get_text

class WinRateGraph(QWidget):
    """Simple widget to draw win rate history."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = [] # List of win rates (0.0 to 1.0)
        self.setBackgroundRole(Qt.ItemDataRole.NoRole)
        self.setStyleSheet("background-color: white; border: 1px solid gray;")
        self.setMinimumHeight(150)

    def update_history(self, win_rate):
        self.history.append(win_rate)
        self.update()

    def clear(self):
        self.history = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Draw grid
        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.PenStyle.DashLine))
        painter.drawLine(0, int(h/2), w, int(h/2)) # 50% line

        if not self.history:
            return

        # Draw line
        painter.setPen(QPen(QColor(0, 100, 255), 2))

        step_x = w / max(1, len(self.history) - 1)

        points = []
        for i, rate in enumerate(self.history):
            x = i * step_x
            y = h - (rate * h) # 1.0 is top (0), 0.0 is bottom (h)
            points.append((x, y))

        for i in range(len(points) - 1):
            painter.drawLine(int(points[i][0]), int(points[i][1]),
                             int(points[i+1][0]), int(points[i+1][1]))

class AnalysisTable(QTableWidget):
    """Table to show card statistics."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels([
            get_text("COL_CARD_NAME"),
            get_text("COL_ADOPT_WIN"),
            get_text("COL_PLAY_WIN"),
            get_text("COL_GAMES")
        ])
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.setSortingEnabled(True)

    def update_data(self, adoption_stats, battle_stats, card_db):
        """
        adoption_stats: {card_id: {'wins': int, 'games': int}}
        battle_stats: {card_id: {'play_count': int, 'win_count': int, ...}}
        card_db: {card_id: CardDefinition}
        """
        self.setRowCount(0)

        all_ids = set(adoption_stats.keys()) | set(battle_stats.keys())

        rows = []
        for cid in all_ids:
            name = "Unknown"
            if cid in card_db:
                name = card_db[cid].name

            # Adoption Stats
            a_data = adoption_stats.get(cid, {'wins': 0, 'games': 0})
            a_games = a_data['games']
            a_rate = (a_data['wins'] / a_games * 100.0) if a_games > 0 else 0.0

            # Battle Stats
            b_data = battle_stats.get(cid, {'play_count': 0, 'win_count': 0})
            b_plays = b_data['play_count']
            b_rate = (b_data['win_count'] / b_plays * 100.0) if b_plays > 0 else 0.0

            # Filter: Show only if significant data or high win rate
            if a_games == 0 and b_plays == 0:
                continue

            rows.append((name, a_rate, b_rate, a_games))

        self.setRowCount(len(rows))
        for i, (name, a_rate, b_rate, a_games) in enumerate(rows):
            self.setItem(i, 0, QTableWidgetItem(str(name)))

            item_a = QTableWidgetItem(f"{a_rate:.1f}%")
            if a_rate > 55.0: item_a.setForeground(QColor("blue"))
            elif a_rate < 45.0: item_a.setForeground(QColor("red"))
            self.setItem(i, 1, item_a)

            item_b = QTableWidgetItem(f"{b_rate:.1f}%")
            if b_rate > 55.0: item_b.setForeground(QColor("blue"))
            self.setItem(i, 2, item_b)

            self.setItem(i, 3, QTableWidgetItem(str(a_games)))

class SimulationThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int) # current, total
    finished_signal = pyqtSignal()
    stats_signal = pyqtSignal(dict, dict) # adoption_stats, battle_stats
    win_rate_signal = pyqtSignal(float)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = True
        self.runner = None
        self.adoption_stats = {} # {card_id: {'wins': 0, 'games': 0}}
        self.total_wins = 0
        self.total_matches = 0

    def run(self):
        if dm_ai_module is None:
            self.log_signal.emit("Error: dm_ai_module not found.")
            return

        try:
            # 1. Load Cards
            self.log_signal.emit("Loading cards...")
            loader = dm_ai_module.JsonLoader()
            card_db = loader.load_cards("data/cards.json")

            # Initialize Stats
            dm_ai_module.initialize_card_stats(dm_ai_module.GameState(1), card_db, 2000) # Arbitrary max ID

            # 2. Setup Runner
            self.runner = dm_ai_module.ParallelRunner(card_db)

            episodes = int(self.config.get('episodes', 100))
            threads = int(self.config.get('threads', 4))
            batch_size = int(self.config.get('batch_size', 32))
            mcts_sims = int(self.config.get('mcts_sims', 800))

            mode = self.config.get('mode', '1v1')
            pbt_enabled = self.config.get('pbt', False)

            # 3. Determine Matchups
            matchups = []

            if mode == '1v1':
                d1 = self.config.get('deck1', [])
                d2 = self.config.get('deck2', [])
                # Repeat the same matchup
                for _ in range(episodes):
                    matchups.append((d1, d2))

            elif mode == 'league':
                deck_pool = self.config.get('deck_pool', []) # List of (name, deck_list)
                if len(deck_pool) < 2:
                    self.log_signal.emit("Error: Need at least 2 decks for league.")
                    return

                # Round Robin
                import itertools
                pairs = list(itertools.combinations(deck_pool, 2))
                # Distribute episodes across pairs
                per_pair = max(1, episodes // len(pairs))

                for _ in range(per_pair):
                    for (n1, d1), (n2, d2) in pairs:
                        matchups.append((d1, d2))

            # Shuffle matchups to mix execution
            random.shuffle(matchups)
            if mode == 'league':
                # Limit total episodes if rounding made it too large
                matchups = matchups[:episodes]

            total_tasks = len(matchups)
            self.log_signal.emit(f"Starting simulation: {total_tasks} matches ({mode}). PBT: {pbt_enabled}")

            # 4. Execution Loop (Batched for memory/GUI updates)
            current_idx = 0
            chunk_size = 10  # Update GUI every 10 matches

            while current_idx < total_tasks and self.running:
                chunk_end = min(current_idx + chunk_size, total_tasks)
                batch_wins = 0

                for i in range(current_idx, chunk_end):
                    if not self.running: break

                    d1, d2 = matchups[i]

                    # Run single match (using play_deck_matchup with 1 game per call for simplicity in tracking)
                    # Note: play_deck_matchup returns win rate for P1 (0.0 to 1.0)
                    # For PBT/League, we might want parallel execution of the CHUNK.
                    # But play_deck_matchup is already parallel if episodes > 1.
                    # Here we call it with episodes=1 to control the loop ourselves.

                    # To use the C++ threading effectively, we should pass a batch.
                    # But play_deck_matchup runs 'N' games.
                    # Let's run 1 game per step to track detailed stats in Python or run 'chunk_size' games.

                    # Actually, play_deck_matchup(episodes, threads, mcts_sims, d1, d2)
                    # Let's run 1 game at a time to allow interruption and progress,
                    # OR run small batches.

                    # Optimization: If 1v1, we can just call it once with all episodes?
                    # But we need intermediate updates for the graph.
                    # So let's run in small batches of 'threads' size.

                    games_to_run = 1
                    # Note: For League, d1/d2 change. For 1v1 they are constant.
                    # Using threads=threads for the runner call.

                    wr = self.runner.play_deck_matchup(games_to_run, threads, batch_size, mcts_sims, d1, d2)

                    # Basic Result Tracking
                    # wr is win rate of d1. If wr > 0.5, d1 won.
                    # Since we run 1 game, wr is 0.0 or 1.0 (draws?)

                    winner_deck = d1 if wr >= 0.5 else d2 # Simplified
                    loser_deck = d2 if wr >= 0.5 else d1

                    # Update Adoption Stats
                    for card_id in winner_deck:
                        if card_id not in self.adoption_stats: self.adoption_stats[card_id] = {'wins':0, 'games':0}
                        self.adoption_stats[card_id]['wins'] += 1
                        self.adoption_stats[card_id]['games'] += 1

                    for card_id in loser_deck:
                        if card_id not in self.adoption_stats: self.adoption_stats[card_id] = {'wins':0, 'games':0}
                        self.adoption_stats[card_id]['games'] += 1

                    if wr >= 0.5:
                        batch_wins += 1
                        if mode == '1v1': self.total_wins += 1

                current_idx = chunk_end
                self.total_matches += (chunk_end - (current_idx - chunk_size))

                # Emit Progress
                self.progress_signal.emit(current_idx, total_tasks)

                # Emit Graph Data (Win Rate for P1 or Global Win Rate)
                current_wr = 0.0
                if self.total_matches > 0:
                    current_wr = self.total_wins / self.total_matches
                self.win_rate_signal.emit(current_wr)

                # Memory Management
                if current_idx % 50 == 0:
                    gc.collect()

            # End of Loop

            # Fetch Battle Zone Stats
            raw_stats = dm_ai_module.get_card_stats()
            # raw_stats is {id: {play_count, win_count, ...}}

            self.stats_signal.emit(self.adoption_stats, raw_stats)
            self.finished_signal.emit()
            self.log_signal.emit(get_text("MSG_SIM_FINISHED"))

        except Exception as e:
            self.log_signal.emit(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.finished_signal.emit()

    def stop(self):
        self.running = False

class SimulationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(get_text("TITLE_SIMULATION"))
        self.resize(1000, 700)

        self.thread = None
        self.card_db = {}
        self.deck_pool = [] # List of (name, deck_list)

        self.init_ui()
        self.load_initial_data()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left Panel: Settings
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(300)

        # Group: Configuration
        grp_settings = QGroupBox(get_text("GRP_SETTINGS"))
        form_layout = QVBoxLayout()

        # Mode
        form_layout.addWidget(QLabel(get_text("SIM_MODE")))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem(get_text("SIM_MODE_1V1"), "1v1")
        self.combo_mode.addItem(get_text("SIM_MODE_LEAGUE"), "league")
        self.combo_mode.currentIndexChanged.connect(self.on_mode_changed)
        form_layout.addWidget(self.combo_mode)

        # PBT
        self.chk_pbt = QCheckBox(get_text("SIM_PBT_ENABLE"))
        form_layout.addWidget(self.chk_pbt)

        # Params
        form_layout.addWidget(QLabel(get_text("LBL_EPISODES")))
        self.edit_episodes = QLineEdit("100")
        form_layout.addWidget(self.edit_episodes)

        form_layout.addWidget(QLabel(get_text("LBL_THREADS")))
        self.edit_threads = QLineEdit("4")
        form_layout.addWidget(self.edit_threads)

        form_layout.addWidget(QLabel(get_text("LBL_BATCH_SIZE")))
        self.edit_batch = QLineEdit("32")
        form_layout.addWidget(self.edit_batch)

        form_layout.addWidget(QLabel(get_text("LBL_MCTS_SIMS")))
        self.edit_sims = QLineEdit("800")
        form_layout.addWidget(self.edit_sims)

        grp_settings.setLayout(form_layout)
        left_layout.addWidget(grp_settings)

        # Group: Deck Selection
        self.grp_decks = QGroupBox("Decks")
        deck_layout = QVBoxLayout()

        # 1v1 Widgets
        self.widget_1v1 = QWidget()
        l_1v1 = QVBoxLayout()
        l_1v1.setContentsMargins(0,0,0,0)
        l_1v1.addWidget(QLabel(get_text("LBL_DECK_1")))
        self.btn_deck1 = QPushButton(get_text("BTN_SELECT_FILE"))
        self.btn_deck1.clicked.connect(lambda: self.select_deck(1))
        self.lbl_deck1_path = QLabel("None")
        l_1v1.addWidget(self.btn_deck1)
        l_1v1.addWidget(self.lbl_deck1_path)

        l_1v1.addWidget(QLabel(get_text("LBL_DECK_2")))
        self.btn_deck2 = QPushButton(get_text("BTN_SELECT_FILE"))
        self.btn_deck2.clicked.connect(lambda: self.select_deck(2))
        self.lbl_deck2_path = QLabel("None")
        l_1v1.addWidget(self.btn_deck2)
        l_1v1.addWidget(self.lbl_deck2_path)
        self.widget_1v1.setLayout(l_1v1)
        deck_layout.addWidget(self.widget_1v1)

        # League Widgets
        self.widget_league = QWidget()
        l_league = QVBoxLayout()
        l_league.setContentsMargins(0,0,0,0)
        self.btn_deck_folder = QPushButton(get_text("SIM_DECK_FOLDER"))
        self.btn_deck_folder.clicked.connect(self.select_deck_folder)
        l_league.addWidget(self.btn_deck_folder)
        self.lbl_deck_count = QLabel("0 Decks selected")
        l_league.addWidget(self.lbl_deck_count)
        self.widget_league.setLayout(l_league)
        self.widget_league.setVisible(False)
        deck_layout.addWidget(self.widget_league)

        self.grp_decks.setLayout(deck_layout)
        left_layout.addWidget(self.grp_decks)

        # Buttons
        self.btn_run = QPushButton(get_text("BTN_RUN"))
        self.btn_run.clicked.connect(self.start_simulation)
        self.btn_run.setStyleSheet("font-weight: bold; padding: 10px;")
        left_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton(get_text("BTN_STOP"))
        self.btn_stop.clicked.connect(self.stop_simulation)
        self.btn_stop.setEnabled(False)
        left_layout.addWidget(self.btn_stop)

        left_layout.addStretch()

        # Right Panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        self.tabs = QTabWidget()

        # Tab 1: Log & Graph
        tab_log = QWidget()
        l_log = QVBoxLayout()
        self.graph = WinRateGraph()
        l_log.addWidget(QLabel("Win Rate History"))
        l_log.addWidget(self.graph)
        l_log.addWidget(QLabel("Log"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        l_log.addWidget(self.log_text)
        self.progress_bar = QProgressBar()
        l_log.addWidget(self.progress_bar)
        tab_log.setLayout(l_log)
        self.tabs.addTab(tab_log, get_text("SIM_TAB_LOG"))

        # Tab 2: Analysis
        tab_analysis = QWidget()
        l_analysis = QVBoxLayout()
        self.analysis_table = AnalysisTable()
        l_analysis.addWidget(self.analysis_table)
        tab_analysis.setLayout(l_analysis)
        self.tabs.addTab(tab_analysis, get_text("SIM_TAB_ANALYSIS"))

        right_layout.addWidget(self.tabs)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.deck1_data = []
        self.deck2_data = []

    def load_initial_data(self):
        if dm_ai_module:
            loader = dm_ai_module.JsonLoader()
            self.card_db = loader.load_cards("data/cards.json")

    def on_mode_changed(self):
        mode = self.combo_mode.currentData()
        is_league = (mode == "league")
        self.widget_1v1.setVisible(not is_league)
        self.widget_league.setVisible(is_league)

    def select_deck(self, slot):
        path, _ = QFileDialog.getOpenFileName(self, "Select Deck", "data/decks", "Text Files (*.txt);;All Files (*)")
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read().strip().split('\n')
                deck = [int(x) for x in content if x.strip().isdigit()]

                if slot == 1:
                    self.lbl_deck1_path.setText(os.path.basename(path))
                    self.deck1_data = deck
                else:
                    self.lbl_deck2_path.setText(os.path.basename(path))
                    self.deck2_data = deck
            except Exception as e:
                self.log(f"Error loading deck: {e}")

    def select_deck_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Deck Folder", "data/decks")
        if folder:
            self.deck_pool = []
            files = [f for f in os.listdir(folder) if f.endswith('.txt')]
            for f in files:
                try:
                    full_path = os.path.join(folder, f)
                    with open(full_path, 'r', encoding='utf-8') as fh:
                        content = fh.read().strip().split('\n')
                    deck = [int(x) for x in content if x.strip().isdigit()]
                    if deck:
                        self.deck_pool.append((f, deck))
                except:
                    pass
            self.lbl_deck_count.setText(f"{len(self.deck_pool)} Decks loaded")

    def log(self, message):
        self.log_text.append(message)

    def start_simulation(self):
        config = {
            'episodes': self.edit_episodes.text(),
            'threads': self.edit_threads.text(),
            'batch_size': self.edit_batch.text(),
            'mcts_sims': self.edit_sims.text(),
            'mode': self.combo_mode.currentData(),
            'pbt': self.chk_pbt.isChecked(),
            'deck1': self.deck1_data,
            'deck2': self.deck2_data,
            'deck_pool': self.deck_pool
        }

        # Validation
        if config['mode'] == '1v1' and (not self.deck1_data or not self.deck2_data):
            self.log("Error: Select both decks for 1v1.")
            return
        if config['mode'] == 'league' and len(self.deck_pool) < 2:
            self.log("Error: Need at least 2 decks for league.")
            return

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_text.clear()
        self.graph.clear()
        self.analysis_table.setRowCount(0)
        self.log(get_text("MSG_SIM_START"))

        self.thread = SimulationThread(config)
        self.thread.log_signal.connect(self.log)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.win_rate_signal.connect(self.graph.update_history)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.start()

    def stop_simulation(self):
        if self.thread:
            self.thread.stop()
            self.log(get_text("MSG_SIM_STOPPED"))
            self.btn_stop.setEnabled(False)

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def update_stats(self, adoption, battle):
        self.analysis_table.update_data(adoption, battle, self.card_db)

    def on_finished(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
