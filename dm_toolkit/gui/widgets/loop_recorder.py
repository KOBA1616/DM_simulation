# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QTextEdit
)
from PyQt6.QtCore import QTimer
import dm_ai_module
from dm_toolkit.gui.i18n import tr

class LoopRecorderWidget(QWidget):
    def __init__(self, game_state_ref, parent=None):
        super().__init__(parent)
        self.gs_ref = game_state_ref # Reference to GameWindow's GS wrapper (or we pass it in update)
        self._layout = QVBoxLayout(self)

        self.status_label = QLabel(tr("Ready"))
        self._layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton(tr("Start Recording"))
        self.start_btn.clicked.connect(self.start_recording)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton(tr("Stop & Verify"))
        self.stop_btn.clicked.connect(self.stop_and_verify)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)

        self._layout.addLayout(btn_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self._layout.addWidget(self.log_text)

        self.recording = False
        self.start_hash = 0
        self.start_turn = 0
        self.action_history = []

        # Initial resources
        self.initial_hand_size = 0
        self.initial_mana_size = 0
        self.initial_shield_count = 0

    def start_recording(self):
        if not self.gs_ref or not self.gs_ref():
            return

        gs = self.gs_ref()
        self.start_hash = gs.calculate_hash()
        self.start_turn = gs.turn_number
        self.recording = True

        p0 = gs.players[0]
        self.initial_hand_size = len(p0.hand)
        self.initial_mana_size = len(p0.mana_zone)
        self.initial_shield_count = len(p0.shield_zone)

        self.action_history = []

        self.status_label.setText(tr("Recording Loop..."))
        self.log_text.clear()
        self.log_text.append(tr("Start Hash: {hash}").format(hash=self.start_hash))
        self.log_text.append(
            tr("Resources: Hand={hand}, Mana={mana}").format(
                hand=self.initial_hand_size,
                mana=self.initial_mana_size,
            )
        )

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def record_action(self, action_str):
        if self.recording:
            self.action_history.append(action_str)
            self.log_text.append(tr("Action: {action}").format(action=action_str))

    def stop_and_verify(self):
        if not self.gs_ref or not self.gs_ref():
            return

        self.recording = False
        gs = self.gs_ref()
        current_hash = gs.calculate_hash()

        p0 = gs.players[0]
        current_hand = len(p0.hand)
        current_mana = len(p0.mana_zone)

        self.log_text.append(tr("End Hash: {hash}").format(hash=current_hash))

        if current_hash == self.start_hash:
               self.log_text.append(tr("State Match: YES (Exact Hash)"))
        else:
               self.log_text.append(tr("State Match: NO"))

        # Check advantage
        diff_hand = current_hand - self.initial_hand_size
        diff_mana = current_mana - self.initial_mana_size

        self.log_text.append(tr("Hand Diff: {diff}").format(diff=diff_hand))
        self.log_text.append(tr("Mana Diff: {diff}").format(diff=diff_mana))

        if current_hash == self.start_hash:
            if diff_hand > 0 or diff_mana > 0:
                self.log_text.append(tr("RESULT: Infinite Loop with Advantage Proven!"))
            else:
                 self.log_text.append(tr("RESULT: Loop Proven (No Resource Gain detected yet)"))

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(tr("Ready"))
