# -*- coding: utf-8 -*-
"""
方針B: アクション選択パネル
人間プレイヤーがカードをクリックした際に関連コマンドを表示し、
QInputDialog を使わずにボード上のパネルで操作できるようにする。
"""
from typing import Any, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QPushButton, QLabel, QListWidgetItem
)
from PyQt6.QtCore import pyqtSignal, Qt

from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.command_describer import describe_command


class ActionPanel(QWidget):
    """人間プレイヤー向けアクション選択パネル（方針B）。

    カードをクリックすると関連コマンドがリスト表示される。
    ダブルクリック / Executeボタンでコマンドを実行する。
    """
    command_selected = pyqtSignal(object)  # emit: CommandDef

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(180)
        self._pending_commands: List[Any] = []
        self._gs: Any = None
        self._card_db: Any = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # タイトル
        self.title_label = QLabel(tr("Action Panel"))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            "font-weight: bold; font-size: 11px; "
            "background-color: #2c3e50; color: white; padding: 4px; border-radius: 3px;"
        )
        layout.addWidget(self.title_label)

        # 状態メッセージ
        self.status_label = QLabel(tr("— Select a card —"))
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "color: #aaa; font-size: 10px; padding: 2px;"
        )
        layout.addWidget(self.status_label)

        # コマンドリスト
        self.cmd_list = QListWidget()
        self.cmd_list.setStyleSheet(
            "QListWidget { font-size: 11px; }"
            "QListWidget::item { padding: 4px; }"
            "QListWidget::item:selected { background-color: #3498db; color: white; }"
            "QListWidget::item:hover { background-color: #ecf0f1; color: #333; }"
        )
        self.cmd_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.cmd_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.cmd_list, stretch=1)

        # 実行ボタン
        self.execute_btn = QPushButton(tr("Execute"))
        self.execute_btn.setEnabled(False)
        self.execute_btn.setStyleSheet(
            "background-color: #27ae60; color: white; font-weight: bold; "
            "padding: 5px; border-radius: 3px;"
        )
        self.execute_btn.clicked.connect(self._on_execute_clicked)
        layout.addWidget(self.execute_btn)

    # ---- 公開API ----

    def set_commands(self, commands: List[Any], gs: Any, card_db: Any):
        """コマンドリストを更新して表示する（方針B）。"""
        self._pending_commands = list(commands)
        self._gs = gs
        self._card_db = card_db
        self.cmd_list.clear()
        self.execute_btn.setEnabled(False)

        if not commands:
            self.status_label.setText(tr("No actions available"))
            self.status_label.setVisible(True)
            return

        self.status_label.setVisible(False)
        for cmd in commands:
            try:
                desc = describe_command(cmd, gs, card_db)
            except Exception:
                try:
                    desc = str(cmd.to_dict().get('type', str(cmd)))
                except Exception:
                    desc = str(cmd)
            item = QListWidgetItem(desc)
            self.cmd_list.addItem(item)

    def clear_commands(self):
        """コマンドリストをクリアして初期状態に戻す。"""
        self._pending_commands = []
        self.cmd_list.clear()
        self.execute_btn.setEnabled(False)
        self.status_label.setText(tr("— Select a card —"))
        self.status_label.setVisible(True)

    def set_waiting_target_mode(self, waiting: bool, msg: str = ""):
        """SELECT_TARGET 待機中モード（対象を選んでから確定ボタンを押す操作フロー）。"""
        if waiting:
            self.clear_commands()
            if msg:
                self.status_label.setText(msg)
            self.execute_btn.setVisible(False)
        else:
            self.execute_btn.setVisible(True)

    # ---- 内部ハンドラ ----

    def _on_item_clicked(self, _item: QListWidgetItem):
        self.execute_btn.setEnabled(True)

    def _on_item_double_clicked(self, _item: QListWidgetItem):
        self._execute_selected()

    def _on_execute_clicked(self):
        self._execute_selected()

    def _execute_selected(self):
        row = self.cmd_list.currentRow()
        if 0 <= row < len(self._pending_commands):
            cmd = self._pending_commands[row]
            self.clear_commands()
            self.command_selected.emit(cmd)
