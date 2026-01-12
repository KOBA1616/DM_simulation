from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel,
    QTabWidget, QSplitter, QTableWidget, QTableWidgetItem,
    QCheckBox, QPushButton, QComboBox, QGroupBox, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.i18n import tr
from dm_toolkit.engine.compat import EngineCompat

class CardEffectDebugger(QWidget):
    """
    Debug widget for card effects.
    Features:
    - Execution Trace: Shows history of resolved commands/effects.
    - Variable Watcher: Shows simulated or actual variables (if available).
    - Pending Stack: Detailed view of pending effects.
    """

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # Tab 1: Execution Trace
        self.trace_tab = QWidget()
        self.trace_layout = QVBoxLayout(self.trace_tab)
        self.trace_list = QListWidget()
        self.trace_layout.addWidget(QLabel(tr("Effect Execution History")))
        self.trace_layout.addWidget(self.trace_list)
        self.tabs.addTab(self.trace_tab, tr("Trace"))

        # Tab 2: Stack & Variables
        self.state_tab = QWidget()
        self.state_layout = QVBoxLayout(self.state_tab)

        # Pending Effects Table
        self.state_layout.addWidget(QLabel(tr("Pending Effects Queue")))
        self.pending_table = QTableWidget()
        self.pending_table.setColumnCount(4)
        self.pending_table.setHorizontalHeaderLabels([tr("Type"), tr("Source"), tr("Controller"), tr("Details")])
        header = self.pending_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.state_layout.addWidget(self.pending_table)

        # Variable Watcher (Placeholder for now until Context is exposed)
        self.state_layout.addWidget(QLabel(tr("Variable Watcher (Mock)")))
        self.var_table = QTableWidget()
        self.var_table.setColumnCount(2)
        self.var_table.setHorizontalHeaderLabels([tr("Name"), tr("Value")])
        self.var_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.state_layout.addWidget(self.var_table)

        self.tabs.addTab(self.state_tab, tr("State"))

        # Tab 3: Breakpoints (Placeholder)
        self.break_tab = QWidget()
        self.break_layout = QVBoxLayout(self.break_tab)
        self.break_check = QCheckBox(tr("Enable Breakpoints"))
        self.break_layout.addWidget(self.break_check)
        self.break_list = QListWidget()
        self.break_layout.addWidget(QLabel(tr("Breakpoints (Not Implemented)")))
        self.break_layout.addWidget(self.break_list)
        self.tabs.addTab(self.break_tab, tr("Breakpoints"))

        self.last_history_len = 0

    def update_state(self, game_state, card_db):
        """Updates the view based on current game state."""
        if not game_state:
            return

        # Update Trace
        history = EngineCompat.get_command_history(game_state)
        current_len = len(history)
        if current_len > self.last_history_len:
            for i in range(self.last_history_len, current_len):
                cmd = history[i]
                # Simple string representation for now
                self.trace_list.addItem(str(cmd))
            self.trace_list.scrollToBottom()
            self.last_history_len = current_len

        # Update Pending Effects
        pending_info = EngineCompat.get_pending_effects_info(game_state)
        self.pending_table.setRowCount(len(pending_info))
        for i, (p_type, source_id, controller) in enumerate(pending_info):
            self.pending_table.setItem(i, 0, QTableWidgetItem(str(p_type)))

            # Resolve source name
            source_name = f"Instance {source_id}"
            try:
                inst = game_state.get_card_instance(source_id)
                card_def = card_db.get(inst.card_id)
                if card_def:
                    source_name = card_def.name
            except:
                pass

            self.pending_table.setItem(i, 1, QTableWidgetItem(source_name))
            self.pending_table.setItem(i, 2, QTableWidgetItem(f"P{controller}"))
            self.pending_table.setItem(i, 3, QTableWidgetItem("-")) # Details need more exposure

        # Update Variables (Placeholder)
        # We don't have direct access to execution_context from python bindings yet.
        # This would require C++ changes.
        self.var_table.setRowCount(0)
