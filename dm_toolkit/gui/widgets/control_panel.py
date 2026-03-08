# -*- coding: utf-8 -*-
from typing import Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QRadioButton, QButtonGroup, QCheckBox, QLabel, QListWidget
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

    # アクションリストでコマンドが選択・実行されたときのシグナル（ActionPanel 移行）
    action_command_selected = pyqtSignal(object)

    # Signals for other tools (connected to main window actions usually, but we can expose them)
    deck_builder_clicked = pyqtSignal()
    card_editor_clicked = pyqtSignal()
    batch_sim_clicked = pyqtSignal()

    # Signals for Load Deck
    load_deck_p0_clicked = pyqtSignal()
    load_deck_p1_clicked = pyqtSignal()

    # Signal for God View
    god_view_toggled = pyqtSignal(bool)

    # Setup signals
    # 再発防止: setup_clicked (セットアップ実行) と setup_config_clicked (設定ダイアログ) は必ず分けること。
    setup_clicked = pyqtSignal()
    setup_config_clicked = pyqtSignal()

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

        # --- セットアップグループ
        self.setup_group = QGroupBox(tr("Setup"))
        setup_layout = QHBoxLayout()

        self.setup_config_btn = QPushButton(tr("Setup Config"))
        self.setup_config_btn.setToolTip(tr("Configure decks and setup options"))
        self.setup_config_btn.clicked.connect(self.setup_config_clicked.emit)
        setup_layout.addWidget(self.setup_config_btn)

        self.setup_btn = QPushButton(tr("Setup"))
        self.setup_btn.setToolTip(tr("Shuffle decks and place cards (deck / shields / hand)"))
        self.setup_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.setup_btn.setShortcut("Ctrl+Shift+S")
        self.setup_btn.clicked.connect(self.setup_clicked.emit)
        setup_layout.addWidget(self.setup_btn)

        self.setup_group.setLayout(setup_layout)
        self.layout_main.addWidget(self.setup_group)

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

        # --- P0 Human Controls（人間プレイヤー操作パネル）
        self.p0_control_group = QGroupBox(tr("P0 Controls"))
        p0_ctrl_layout = QVBoxLayout()
        p0_ctrl_layout.setSpacing(4)
        p0_ctrl_layout.setContentsMargins(4, 6, 4, 6)

        # 状態/ヒントラベル
        self.p0_status_label = QLabel(tr("— Select a card —"))
        self.p0_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.p0_status_label.setWordWrap(True)
        self.p0_status_label.setStyleSheet(
            "color: #888; font-size: 10px; padding: 2px 4px;"
        )
        p0_ctrl_layout.addWidget(self.p0_status_label)

        # アクション選択リスト
        self.action_list = QListWidget()
        self.action_list.setMinimumHeight(80)
        self.action_list.setStyleSheet(
            "QListWidget { font-size: 11px; border: 1px solid #ccc; border-radius: 3px;}"
            "QListWidget::item { padding: 5px; }"
            "QListWidget::item:selected { background-color: #3498db; color: white; }"
            "QListWidget::item:hover { background-color: #dce9f5; color: #333; }"
        )
        self.action_list.itemDoubleClicked.connect(self._on_action_double_clicked)
        self.action_list.itemClicked.connect(self._on_action_clicked)
        p0_ctrl_layout.addWidget(self.action_list, stretch=1)

        # 実行ボタン
        self.action_execute_btn = QPushButton(tr("Execute"))
        self.action_execute_btn.setEnabled(False)
        self.action_execute_btn.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 6px; border-radius: 3px;"
        )
        self.action_execute_btn.clicked.connect(self._on_action_execute)
        p0_ctrl_layout.addWidget(self.action_execute_btn)

        # パス/ターン終了 + 確定（横並び）
        op_row = QHBoxLayout()
        self.p0_ctrl_pass = QPushButton(tr("Pass / End"))
        self.p0_ctrl_pass.setShortcut("Ctrl+E")
        self.p0_ctrl_pass.clicked.connect(self.pass_clicked.emit)
        self.p0_ctrl_pass.setEnabled(False)
        self.p0_ctrl_pass.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; border-radius: 3px;"
        )
        op_row.addWidget(self.p0_ctrl_pass)

        self.p0_ctrl_confirm = QPushButton(tr("Confirm"))
        self.p0_ctrl_confirm.setShortcut("Return")
        self.p0_ctrl_confirm.clicked.connect(self.confirm_clicked.emit)
        self.p0_ctrl_confirm.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; border-radius: 3px;"
        )
        self.p0_ctrl_confirm.setVisible(False)
        op_row.addWidget(self.p0_ctrl_confirm)
        p0_ctrl_layout.addLayout(op_row)

        # リセット
        self.p0_ctrl_reset = QPushButton(tr("Reset"))
        self.p0_ctrl_reset.setShortcut("Ctrl+R")
        self.p0_ctrl_reset.clicked.connect(self.reset_clicked.emit)
        p0_ctrl_layout.addWidget(self.p0_ctrl_reset)

        self.p0_control_group.setLayout(p0_ctrl_layout)
        self.layout_main.addWidget(self.p0_control_group)

        self.layout_main.addStretch()

        # 内部状態（アクションリスト）
        # 再発防止: action_panel (game_board 右側) を廃止しここで管理
        self._action_pending_commands: list = []
        self._action_gs: Any = None
        self._action_card_db: Any = None

        self._update_p0_controls_visibility()

    def _update_p0_controls_visibility(self):
        show = self.p0_human_radio.isChecked()
        self.p0_control_group.setVisible(show)

    def set_start_button_text(self, text: str):
        self.start_btn.setText(text)

    def set_pass_button_visible(self, visible: bool):
        self.pass_btn.setVisible(visible)
        # P0 操作パネルのパスボタンは有効/無効で制御（表示は維持でレイアウト移動防止）
        if hasattr(self, 'p0_ctrl_pass'):
            self.p0_ctrl_pass.setEnabled(bool(visible))

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
            confirm_text = f"{tr('Confirm')} ({selected_count}/{min_targets}-{max_targets})"

            self.set_confirm_button_text(confirm_text)
            self.set_confirm_button_visible(True)
            self.set_confirm_button_enabled(selected_count >= min_targets)
            # P0 操作パネルの確定ボタンも同期
            if hasattr(self, 'p0_ctrl_confirm'):
                self.p0_ctrl_confirm.setText(confirm_text)
                self.p0_ctrl_confirm.setVisible(True)
                self.p0_ctrl_confirm.setEnabled(selected_count >= min_targets)
        else:
            self.set_confirm_button_visible(False)
            if hasattr(self, 'p0_ctrl_confirm'):
                self.p0_ctrl_confirm.setVisible(False)

    # ---- P0 アクションリスト API（ActionPanel 移行）----

    def set_action_commands(self, commands: list, gs: Any, card_db: Any) -> None:
        """カードクリック時のアクション候補リストを更新する。"""
        from dm_toolkit.gui.utils.command_describer import describe_command
        self._action_pending_commands = list(commands)
        self._action_gs = gs
        self._action_card_db = card_db
        self.action_list.clear()
        self.action_execute_btn.setEnabled(False)
        if not commands:
            self.p0_status_label.setText(tr("No actions available"))
            self.p0_status_label.setVisible(True)
            return
        self.p0_status_label.setVisible(False)
        for cmd in commands:
            try:
                desc = describe_command(cmd, gs, card_db)
            except Exception:
                try:
                    desc = str(cmd.to_dict().get('type', str(cmd)))
                except Exception:
                    desc = str(cmd)
            self.action_list.addItem(desc)

    def clear_action_commands(self) -> None:
        """アクションリストをクリアして初期状態に戻す。"""
        self._action_pending_commands = []
        self.action_list.clear()
        self.action_execute_btn.setEnabled(False)
        self.p0_status_label.setText(tr("— Select a card —"))
        self.p0_status_label.setVisible(True)

    def set_action_waiting_mode(self, waiting: bool, msg: str = "") -> None:
        """SELECT_TARGET 待機中モード: リストを無効化し案内メッセージを表示する。"""
        if waiting:
            self.clear_action_commands()
            if msg:
                self.p0_status_label.setText(msg)
            self.action_execute_btn.setVisible(False)
        else:
            self.action_execute_btn.setVisible(True)

    def _on_action_clicked(self, _item) -> None:
        self.action_execute_btn.setEnabled(True)

    def _on_action_double_clicked(self, _item) -> None:
        self._execute_selected_action()

    def _on_action_execute(self) -> None:
        self._execute_selected_action()

    def _execute_selected_action(self) -> None:
        row = self.action_list.currentRow()
        if 0 <= row < len(self._action_pending_commands):
            cmd = self._action_pending_commands[row]
            self.clear_action_commands()
            self.action_command_selected.emit(cmd)
