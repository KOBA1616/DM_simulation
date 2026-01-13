# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox
from dm_toolkit.gui.i18n import tr
from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit.types import GameState

class GameStatusWidget(QWidget):
    """
    Displays the current game status (Turn, Phase, Active Player).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)

        self.group = QGroupBox(tr("Game Status"))
        top_layout = QVBoxLayout()
        status_layout = QHBoxLayout()

        self.turn_label = QLabel(tr("Turn: {turn}").format(turn="?"))
        self.turn_label.setStyleSheet("font-weight: bold;")
        self.phase_label = QLabel(tr("Phase: {phase}").format(phase="?"))
        self.active_label = QLabel(tr("Active: P{player_id}").format(player_id="?"))

        status_layout.addWidget(self.turn_label)
        status_layout.addWidget(self.phase_label)
        status_layout.addWidget(self.active_label)
        top_layout.addLayout(status_layout)

        self.group.setLayout(top_layout)
        self.layout_main.addWidget(self.group)

    def update_state(self, gs: GameState) -> None:
        if gs is None:
            return

        turn_number = EngineCompat.get_turn_number(gs)
        current_phase = EngineCompat.get_current_phase(gs)

        phase_map = {
            "START": "Start Phase",
            "DRAW": "Draw Phase",
            "MANA": "Mana Phase",
            "MAIN": "Main Phase",
            "ATTACK": "Attack Phase",
            "BLOCK": "Block Phase",
            "END": "End Phase"
        }
        phase_key = phase_map.get(str(current_phase), str(current_phase))

        active_pid = EngineCompat.get_active_player_id(gs)
        self.turn_label.setText(tr("Turn: {turn}").format(turn=turn_number))
        self.phase_label.setText(tr("Phase: {phase}").format(phase=tr(phase_key)))
        self.active_label.setText(tr("Active: P{player_id}").format(player_id=active_pid))
