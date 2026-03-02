# -*- coding: utf-8 -*-
"""
SetupConfigDialog: ゲームセットアップのデッキ設定ダイアログ。

デッキ選択は「デッキ読み込みボタン」と同じ方式：
  QFileDialog で data/decks/ 配下の JSON ファイルを選択する。
ファイル未選択の場合はデフォルト（meta_decks.json の先頭デッキ）が使用される。

今後の拡張: シールド上限、手札枚数、初期マナ等の設定を追加予定。

再発防止:
  - ダイアログは accept/reject で結果を返す。
  - get_config() で設定辞書を返し、do_setup() 側で解釈する。
  - ファイルパスが空の場合は呼び出し側がデフォルトを使う設計とする。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QDialogButtonBox, QWidget, QFileDialog,
    QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt

from dm_toolkit.gui.i18n import tr

# デッキ JSON のデフォルト検索ディレクトリ
_DEFAULT_DECK_DIR = "data/decks"


class _DeckFileSelector(QGroupBox):
    """1プレイヤー分のデッキファイル選択 UI（デッキ読み込みボタンと同方式）。"""

    def __init__(self, player_label: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(player_label, parent)

        layout = QHBoxLayout(self)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(tr("(default deck)"))
        self._path_edit.setReadOnly(True)
        self._path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._path_edit, 1)

        self._browse_btn = QPushButton(tr("Browse..."))
        self._browse_btn.clicked.connect(self._browse)
        layout.addWidget(self._browse_btn)

        self._clear_btn = QPushButton(tr("Clear"))
        self._clear_btn.setToolTip(tr("Use default deck"))
        self._clear_btn.clicked.connect(self._clear)
        layout.addWidget(self._clear_btn)

    def _browse(self) -> None:
        """デッキ読み込みボタンと同じ QFileDialog を使ったファイル選択。"""
        os.makedirs(_DEFAULT_DECK_DIR, exist_ok=True)
        fname, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select Deck JSON"),
            _DEFAULT_DECK_DIR,
            "JSON Files (*.json)",
        )
        if fname:
            self._path_edit.setText(fname)

    def _clear(self) -> None:
        self._path_edit.clear()

    def get_file_path(self) -> str:
        """選択されたファイルパスを返す。空文字はデフォルト使用を意味する。"""
        return self._path_edit.text().strip()

    def set_file_path(self, path: str) -> None:
        self._path_edit.setText(path)


class SetupConfigDialog(QDialog):
    """
    セットアップ設定ダイアログ。

    Usage::

        dlg = SetupConfigDialog(parent=self, current_config=self._setup_config)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._setup_config = dlg.get_config()
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        current_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Setup Config"))
        self.setMinimumWidth(520)

        cfg = current_config or {}
        layout = QVBoxLayout(self)

        # 説明ラベル
        info = QLabel(tr(
            "Select a deck JSON file for each player (same as Load Deck button).\n"
            "Leave blank to use the default deck (meta_decks.json).\n"
            "Setup will shuffle the deck and place shields / hand automatically."
        ))
        info.setWordWrap(True)
        layout.addWidget(info)

        # --- P0 ---
        self._p0_selector = _DeckFileSelector(tr("Player 0 (Self) Deck"))
        self._p0_selector.set_file_path(cfg.get("p0_file_path", ""))
        layout.addWidget(self._p0_selector)

        # --- P1 ---
        self._p1_selector = _DeckFileSelector(tr("Player 1 (Opponent) Deck"))
        self._p1_selector.set_file_path(cfg.get("p1_file_path", ""))
        layout.addWidget(self._p1_selector)

        # --- 将来の拡張エリア（プレースホルダー）---
        future_group = QGroupBox(tr("Advanced (Future Extensions)"))
        fl = QHBoxLayout(future_group)
        fl.addWidget(QLabel(tr("(More options will be added here)")))
        future_group.setEnabled(False)
        layout.addWidget(future_group)

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        """OKボタン: ファイルが指定されている場合は存在確認してから accept。"""
        for sel, name in [(self._p0_selector, "P0"), (self._p1_selector, "P1")]:
            path = sel.get_file_path()
            if path and not os.path.exists(path):
                QMessageBox.warning(
                    self,
                    tr("Validation Error"),
                    tr(f"{name}: File not found: {path}"),
                )
                return
        self.accept()

    def get_config(self) -> Dict[str, Any]:
        """ダイアログの設定を辞書で返す。"""
        return {
            "p0_file_path": self._p0_selector.get_file_path(),
            "p1_file_path": self._p1_selector.get_file_path(),
        }
