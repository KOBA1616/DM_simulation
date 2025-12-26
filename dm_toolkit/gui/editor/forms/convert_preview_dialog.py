# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt
import json
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor import text_generator
from dm_toolkit.gui.editor.text_generator import CardTextGenerator


class ConvertPreviewDialog(QDialog):
    """Shows original Action JSON and converted Command JSON side-by-side.

    Returns user's choice via exec_():
      Accepted -> QDialog.Accepted (use converted)
      Rejected -> QDialog.Rejected (keep action)
      Cancel -> QDialog.DialogCode(0) (cancel save)
    """
    def __init__(self, parent, action_data: dict, converted: dict):
        super().__init__(parent)
        # Safe defaults for attributes used by external checkers/tests
        self.action_text = None
        self.conv_text = None
        self.action_preview = None
        self.conv_preview = None
        self.use_btn = None
        self.keep_btn = None
        self.cancel_btn = None
        self.setWindowTitle(tr("Conversion Preview"))
        self.resize(800, 480)

        layout = QVBoxLayout(self)

        hint = QLabel(tr("The editor attempted to convert this legacy Action to a Command. Review the result."))
        layout.addWidget(hint)

        panes = QHBoxLayout()
        self.action_text = QTextEdit()
        self.action_text.setReadOnly(True)
        self.action_text.setPlainText(json.dumps(action_data, indent=2, ensure_ascii=False))

        self.conv_text = QTextEdit()
        self.conv_text.setReadOnly(True)
        self.conv_text.setPlainText(json.dumps(converted, indent=2, ensure_ascii=False))

        panes.addWidget(self.action_text)
        panes.addWidget(self.conv_text)
        layout.addLayout(panes)

        # Generated human-readable previews
        previews = QHBoxLayout()
        self.action_preview = QTextEdit()
        self.action_preview.setReadOnly(True)
        try:
            a_text = text_generator.CardTextGenerator._format_action(action_data)
        except Exception:
            a_text = ""
        self.action_preview.setPlainText(a_text)

        self.conv_preview = QTextEdit()
        self.conv_preview.setReadOnly(True)
        try:
            # Prefer using _format_command for command-shaped dicts
            c_text = text_generator.CardTextGenerator._format_command(converted)
        except Exception:
            try:
                c_text = text_generator.CardTextGenerator._format_action(converted)
            except Exception:
                c_text = ""
        self.conv_preview.setPlainText(c_text)

        previews.addWidget(self.action_preview)
        previews.addWidget(self.conv_preview)
        layout.addLayout(previews)

        # Sample/stat explanation area
        self.stat_sample = QTextEdit()
        self.stat_sample.setReadOnly(True)
        self.stat_sample.setFixedHeight(80)
        try:
            sample_text = self._build_stat_sample_text(action_data, converted)
        except Exception:
            sample_text = ""
        self.stat_sample.setPlainText(sample_text)
        layout.addWidget(self.stat_sample)

        btns = QHBoxLayout()
        self.use_btn = QPushButton(tr("Use Converted Command"))
        self.keep_btn = QPushButton(tr("Keep as Action"))
        self.cancel_btn = QPushButton(tr("Cancel"))

        self.use_btn.clicked.connect(self.accept)
        self.keep_btn.clicked.connect(self.reject)
        self.cancel_btn.clicked.connect(self.close)

        btns.addWidget(self.use_btn)
        btns.addWidget(self.keep_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

    def keyPressEvent(self, ev):
        # Escape should close as cancel
        if ev.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(ev)

    def _build_stat_sample_text(self, action_data: dict, converted: dict) -> str:
        """Build a short explanatory/sample text for GET_GAME_STAT keys.

        If caller provides a `sample_mana` list (of civilization strings or card dicts),
        compute a concrete example. Otherwise emit a descriptive explanation.
        """
        texts = []
        # Look for stat key in action_data first, then converted
        key = action_data.get('str_val') or action_data.get('str_param') or converted.get('str_param') or converted.get('str_val') or ''
        if not key:
            return ""

        # Human-readable label if known
        stat_name, unit = CardTextGenerator.STAT_KEY_MAP.get(key, (None, None))
        if stat_name:
            texts.append(f"統計キー: {key} → {stat_name}{unit}")
        else:
            texts.append(f"統計キー: {key}")

        # Try to compute example if caller provided sample mana list
        sample = action_data.get('sample_mana') or action_data.get('sample_zone') or converted.get('sample_mana')
        if sample and isinstance(sample, list):
            # Normalize: accept list of civ strings or list of dicts with 'civilizations'
            civs = set()
            for entry in sample:
                if isinstance(entry, str):
                    civs.add(entry)
                elif isinstance(entry, dict):
                    for c in entry.get('civilizations', []):
                        civs.add(c)
            texts.append(f"サンプルマナ: {len(sample)}枚 (文明種類: {', '.join(sorted(civs))})")
            if key == 'MANA_CIVILIZATION_COUNT':
                texts.append(f"計算結果（重複除去）: {len(civs)}{unit}")

        else:
            # Provide descriptive sample for known key
            if key == 'MANA_CIVILIZATION_COUNT':
                texts.append("例: マナゾーン内のカードを調べ、含まれる文明の種類数を返します（同じ文明は1つとして数える）。")

        return "\n".join(texts)
