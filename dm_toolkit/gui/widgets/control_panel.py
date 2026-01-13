# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QRadioButton, QButtonGroup, QCheckBox, QLabel
)
from PyQt6.QtCore import pyqtSignal, Qt
from dm_toolkit.gui.i18n import tr

class ControlPanel(QWidget):
    """
    Manages game control buttons (Start, Step, Reset) and player mode selection (Human/AI).
    """
    # Signals for buttons
    start_simulation_clicked = pyqtSignal()
    step_clicked = pyqtSignal()
    pass_clicked = pyqtSignal()
    confirm_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()

    # Signals for other tools (connected to main window actions usually, but we can expose them)
    deck_builder_clicked = pyqtSignal()
    card_editor_clicked = pyqtSignal()
    batch_sim_clicked = pyqtSignal()

    # Signals for Load Deck
    load_deck_p0_clicked = pyqtSignal()
    load_deck_p1_clicked = pyqtSignal()

    # Signal for God View
    god_view_toggled = pyqtSignal(bool)

    # Help
    help_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)

        # --- Top Game Status is managed by main window or another widget,
        # but the request said "control panel" manages Start/Step/Reset.
        # In app.py these were in "Game Status & Operations" dock.
        # We will create a section for Game Controls.

        # Game Control Buttons Group
        self.game_ctrl_group = QGroupBox(tr("Game Control"))
        game_ctrl_layout = QHBoxLayout()

        self.start_btn = QPushButton(tr("Start Sim"))
        self.start_btn.setShortcut("F5")
        self.start_btn.clicked.connect(self.start_simulation_clicked.emit)
        game_ctrl_layout.addWidget(self.start_btn)

        self.step_btn = QPushButton(tr("Step"))
        self.step_btn.setShortcut("Space")
        self.step_btn.clicked.connect(self.step_clicked.emit)
        game_ctrl_layout.addWidget(self.step_btn)

        self.pass_btn = QPushButton(tr("Pass / End Turn"))
        self.pass_btn.setShortcut("Ctrl+E")
        self.pass_btn.clicked.connect(self.pass_clicked.emit)
        self.pass_btn.setVisible(False)
        self.pass_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        game_ctrl_layout.addWidget(self.pass_btn)

        self.confirm_btn = QPushButton(tr("Confirm Selection"))
        self.confirm_btn.setShortcut("Return")
        self.confirm_btn.clicked.connect(self.confirm_clicked.emit)
        self.confirm_btn.setVisible(False)
        self.confirm_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        game_ctrl_layout.addWidget(self.confirm_btn)

        self.reset_btn = QPushButton(tr("Reset"))
        self.reset_btn.setShortcut("Ctrl+R")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        game_ctrl_layout.addWidget(self.reset_btn)

        self.game_ctrl_group.setLayout(game_ctrl_layout)
        self.layout_main.addWidget(self.game_ctrl_group)

        # --- Player Mode Group
        self.mode_group = QGroupBox(tr("Player Mode"))
        mode_layout = QVBoxLayout()

        self.p0_human_radio = QRadioButton(tr("P0 (Self): Human"))
        self.p0_ai_radio = QRadioButton(tr("P0 (Self): AI"))
        self.p0_ai_radio.setChecked(True)
        self.p0_group = QButtonGroup()
        self.p0_group.addButton(self.p0_human_radio)
        self.p0_group.addButton(self.p0_ai_radio)
        self.p0_human_radio.toggled.connect(self._update_p0_controls_visibility)

        mode_layout.addWidget(self.p0_human_radio)
        mode_layout.addWidget(self.p0_ai_radio)

        self.p1_human_radio = QRadioButton(tr("P1 (Opp): Human"))
        self.p1_ai_radio = QRadioButton(tr("P1 (Opp): AI"))
        self.p1_ai_radio.setChecked(True)
        self.p1_group = QButtonGroup()
        self.p1_group.addButton(self.p1_human_radio)
        self.p1_group.addButton(self.p1_ai_radio)

        mode_layout.addWidget(self.p1_human_radio)
        mode_layout.addWidget(self.p1_ai_radio)
        self.mode_group.setLayout(mode_layout)
        self.layout_main.addWidget(self.mode_group)

        # --- Tools Layout
        tools_group = QGroupBox(tr("Tools"))
        tools_layout = QVBoxLayout()
        self.deck_builder_btn = QPushButton(tr("Deck Builder"))
        self.deck_builder_btn.clicked.connect(self.deck_builder_clicked.emit)
        tools_layout.addWidget(self.deck_builder_btn)

        self.card_editor_btn = QPushButton(tr("Card Editor"))
        self.card_editor_btn.clicked.connect(self.card_editor_clicked.emit)
        tools_layout.addWidget(self.card_editor_btn)

        self.sim_dialog_btn = QPushButton(tr("Batch Simulation"))
        self.sim_dialog_btn.clicked.connect(self.batch_sim_clicked.emit)
        tools_layout.addWidget(self.sim_dialog_btn)
        tools_group.setLayout(tools_layout)
        self.layout_main.addWidget(tools_group)

        # --- Deck Management
        deck_group = QGroupBox(tr("Deck Management"))
        deck_layout = QVBoxLayout()
        self.load_deck_p0_btn = QPushButton(tr("Load Deck P0"))
        self.load_deck_p0_btn.clicked.connect(self.load_deck_p0_clicked.emit)
        deck_layout.addWidget(self.load_deck_p0_btn)

        self.load_deck_p1_btn = QPushButton(tr("Load Deck P1"))
        self.load_deck_p1_btn.clicked.connect(self.load_deck_p1_clicked.emit)
        deck_layout.addWidget(self.load_deck_p1_btn)
        deck_group.setLayout(deck_layout)
        self.layout_main.addWidget(deck_group)

        # --- Options
        self.god_view_check = QCheckBox(tr("God View"))
        self.god_view_check.setChecked(False)
        self.god_view_check.stateChanged.connect(lambda state: self.god_view_toggled.emit(state == Qt.CheckState.Checked.value))
        self.layout_main.addWidget(self.god_view_check)

        self.help_btn = QPushButton(tr("Help / Manual"))
        self.help_btn.clicked.connect(self.help_clicked.emit)
        self.layout_main.addWidget(self.help_btn)

        # --- P0 Human Controls (Redundant/Quick Access)
        self.p0_control_group = QGroupBox(tr("P0 Controls"))
        p0_ctrl_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        self.p0_ctrl_start = QPushButton(tr("Start"))
        self.p0_ctrl_start.clicked.connect(self.start_simulation_clicked.emit)
        row1.addWidget(self.p0_ctrl_start)

        self.p0_ctrl_step = QPushButton(tr("Step"))
        self.p0_ctrl_step.clicked.connect(self.step_clicked.emit)
        row1.addWidget(self.p0_ctrl_step)
        p0_ctrl_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.p0_ctrl_pass = QPushButton(tr("Pass / End"))
        self.p0_ctrl_pass.clicked.connect(self.pass_clicked.emit)
        row2.addWidget(self.p0_ctrl_pass)

        self.p0_ctrl_confirm = QPushButton(tr("Confirm"))
        self.p0_ctrl_confirm.clicked.connect(self.confirm_clicked.emit)
        row2.addWidget(self.p0_ctrl_confirm)
        p0_ctrl_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.p0_ctrl_reset = QPushButton(tr("Reset"))
        self.p0_ctrl_reset.clicked.connect(self.reset_clicked.emit)
        row3.addWidget(self.p0_ctrl_reset)
        p0_ctrl_layout.addLayout(row3)

        self.p0_control_group.setLayout(p0_ctrl_layout)
        self.layout_main.addWidget(self.p0_control_group)

        self.layout_main.addStretch()

        self._update_p0_controls_visibility()

    def _update_p0_controls_visibility(self):
        show = self.p0_human_radio.isChecked()
        self.p0_control_group.setVisible(show)

    def set_start_button_text(self, text: str):
        self.start_btn.setText(text)

    def set_pass_button_visible(self, visible: bool):
        self.pass_btn.setVisible(visible)

    def set_confirm_button_visible(self, visible: bool):
        self.confirm_btn.setVisible(visible)

    def set_confirm_button_text(self, text: str):
        self.confirm_btn.setText(text)

    def set_confirm_button_enabled(self, enabled: bool):
        self.confirm_btn.setEnabled(enabled)

    def is_p0_human(self) -> bool:
        return self.p0_human_radio.isChecked()

    def is_p1_human(self) -> bool:
        return self.p1_human_radio.isChecked()

    def is_god_view(self) -> bool:
        return self.god_view_check.isChecked()

    def update_state(self, can_pass: bool, is_waiting_input: bool, pending_query: any, selected_count: int):
        """Updates the state of control buttons based on game context."""
        from dm_toolkit.gui.i18n import tr

        self.set_pass_button_visible(can_pass)

        if is_waiting_input and pending_query is not None and getattr(pending_query, 'query_type', '') == "SELECT_TARGET":
            params = getattr(pending_query, 'params', {})
            min_targets = params.get('min', 1) if hasattr(params, 'get') else 1
            max_targets = params.get('max', 99) if hasattr(params, 'get') else 99

            self.set_confirm_button_text(f"{tr('Confirm')} ({selected_count}/{min_targets}-{max_targets})")
            self.set_confirm_button_visible(True)
            self.set_confirm_button_enabled(selected_count >= min_targets)
        else:
            self.set_confirm_button_visible(False)
