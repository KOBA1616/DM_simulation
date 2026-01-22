from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel,
    QTabWidget, QSplitter, QTableWidget, QTableWidgetItem,
    QCheckBox, QPushButton, QComboBox, QGroupBox, QHeaderView,
    QToolBar
)
from PyQt6.QtCore import Qt, pyqtSignal
from dm_toolkit.gui.i18n import tr
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.gui.utils.card_helpers import get_card_name

class CardEffectDebugger(QWidget):
    """
    Debug widget for card effects.
    Features:
    - Execution Trace: Shows history of resolved commands/effects.
    - Variable Watcher: Shows simulated or actual variables (if available).
    - Pending Stack: Detailed view of pending effects.
    """

    # Signal to request engine step
    step_requested = pyqtSignal()
    # Signal to request resume
    resume_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # Toolbar for Execution Control
        self.toolbar_layout = QHBoxLayout()
        self.step_btn = QPushButton(tr("Step Over"))
        self.resume_btn = QPushButton(tr("Resume"))
        self.pause_btn = QPushButton(tr("Pause")) # Optional manual pause

        self.step_btn.clicked.connect(self.on_step_clicked)
        self.resume_btn.clicked.connect(self.on_resume_clicked)

        self.toolbar_layout.addWidget(self.step_btn)
        self.toolbar_layout.addWidget(self.resume_btn)
        self.toolbar_layout.addStretch()
        self.layout.addLayout(self.toolbar_layout)

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

        # Variable Watcher
        self.state_layout.addWidget(QLabel(tr("Variable Watcher")))
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

    def on_step_clicked(self):
        self.step_requested.emit()

    def on_resume_clicked(self):
        self.resume_requested.emit()

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
        for i, info in enumerate(pending_info):
            # Info is now (type_str, source_id, controller, command_object)
            p_type, source_id, controller, cmd_obj = info

            self.pending_table.setItem(i, 0, QTableWidgetItem(str(p_type)))

            # Resolve source name
            source_name = f"Instance {source_id}"
            try:
                inst = game_state.get_card_instance(source_id)
                card_def = card_db.get(inst.card_id)
                if card_def:
                    source_name = get_card_name(card_def)
            except:
                pass

            self.pending_table.setItem(i, 1, QTableWidgetItem(source_name))
            self.pending_table.setItem(i, 2, QTableWidgetItem(f"P{controller}"))

            # Details Column
            details_str = EngineCompat.get_command_details(cmd_obj)
            self.pending_table.setItem(i, 3, QTableWidgetItem(details_str))

        # Update Variables
        context_vars = EngineCompat.get_execution_context(game_state)
        self.var_table.setRowCount(len(context_vars))
        for i, (name, value) in enumerate(context_vars.items()):
            self.var_table.setItem(i, 0, QTableWidgetItem(str(name)))
            self.var_table.setItem(i, 1, QTableWidgetItem(str(value)))
