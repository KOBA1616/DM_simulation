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

        # Sample input controls (allow user to provide sample_mana or zone for preview)
        sample_row = QHBoxLayout()
        sample_label = QLabel(tr("Sample JSON (e.g. [\"LIGHT\", \"WATER\"])"))
        self.sample_input = QTextEdit()
        self.sample_input.setFixedHeight(60)
        # Pre-fill if caller provided sample in action_data/converted
        initial_sample = None
        if isinstance(action_data.get('sample_mana'), list):
            initial_sample = action_data.get('sample_mana')
        elif isinstance(converted.get('sample_mana'), list):
            initial_sample = converted.get('sample_mana')
        if initial_sample is not None:
            try:
                self.sample_input.setPlainText(json.dumps(initial_sample, ensure_ascii=False))
            except Exception:
                self.sample_input.setPlainText(str(initial_sample))

        self.apply_sample_btn = QPushButton(tr("Apply Sample"))
        self.clear_sample_btn = QPushButton(tr("Clear Sample"))
        sample_row.addWidget(sample_label)
        sample_row.addWidget(self.sample_input)
        sample_row.addWidget(self.apply_sample_btn)
        sample_row.addWidget(self.clear_sample_btn)
        layout.addLayout(sample_row)

        # Generated human-readable previews
        previews = QHBoxLayout()
        self.action_preview = QTextEdit()
        self.action_preview.setReadOnly(True)
        self.conv_preview = QTextEdit()
        self.conv_preview.setReadOnly(True)

        previews.addWidget(self.action_preview)
        previews.addWidget(self.conv_preview)
        layout.addLayout(previews)

        # Sample/stat explanation area
        self.stat_sample = QTextEdit()
        self.stat_sample.setReadOnly(True)
        self.stat_sample.setFixedHeight(80)
        # Internal sample used for previews
        self._sample = None
        try:
            sample_text = self._build_stat_sample_text(action_data, converted, sample=self._sample)
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

        # Wire sample buttons
        self.apply_sample_btn.clicked.connect(lambda: self._on_apply_sample(action_data, converted))
        self.clear_sample_btn.clicked.connect(lambda: self._on_clear_sample(action_data, converted))

        # Initialize previews
        self._refresh_previews(action_data, converted)

    def keyPressEvent(self, ev):
        # Escape should close as cancel
        if ev.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(ev)

    def _build_stat_sample_text(self, action_data: dict, converted: dict, sample: list = None) -> str:
        """Build a short explanatory/sample text for GET_GAME_STAT keys.

        If caller provides a `sample_mana` list (of civilization strings or card dicts),
        compute a concrete example. Otherwise emit a descriptive explanation.
        """
        texts = []
        # Look for stat key in action_data first, then converted
        # prefer explicit sample passed in via converted or action_data, or use internal sample
        key = action_data.get('str_val') or action_data.get('str_param') or converted.get('str_param') or converted.get('str_val') or ''
        if not key:
            return ""

        # Human-readable label if known
        stat_name, unit = CardTextGenerator.STAT_KEY_MAP.get(key, (None, None))
        if stat_name:
            texts.append(f"統計キー: {key} → {stat_name}{unit}")
        else:
            texts.append(f"統計キー: {key}")

        # Allow caller-provided sample to override embedded samples
        if sample is None:
            # If an explicit sample param present in data, prefer it
            if action_data.get('sample_mana'):
                sample = action_data.get('sample_mana')
            elif action_data.get('sample_zone'):
                sample = action_data.get('sample_zone')
            elif converted.get('sample_mana'):
                sample = converted.get('sample_mana')
            # If still none, use the dialog's sample if any
            if sample is None and hasattr(self, '_sample'):
                sample = self._sample
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

    def _on_apply_sample(self, action_data: dict, converted: dict):
        """Parse sample JSON from input and refresh previews."""
        txt = self.sample_input.toPlainText().strip()
        if not txt:
            self._sample = None
            self.stat_sample.setPlainText("")
            self._refresh_previews(action_data, converted)
            return

        try:
            parsed = json.loads(txt)
            if not isinstance(parsed, list):
                raise ValueError("sample must be a JSON array")
            self._sample = parsed
        except Exception as e:
            self.stat_sample.setPlainText(tr("Invalid sample JSON:") + f" {e}")
            return

        # Refresh previews with sample
        self._refresh_previews(action_data, converted)

    def _on_clear_sample(self, action_data: dict, converted: dict):
        self.sample_input.clear()
        self._sample = None
        self._refresh_previews(action_data, converted)

    def _refresh_previews(self, action_data: dict, converted: dict):
        """Regenerate JSON and human-readable previews, and update stat/sample text."""
        # JSON panes
        try:
            self.action_text.setPlainText(json.dumps(action_data, indent=2, ensure_ascii=False))
        except Exception:
            self.action_text.setPlainText(str(action_data))

        try:
            self.conv_text.setPlainText(json.dumps(converted, indent=2, ensure_ascii=False))
        except Exception:
            self.conv_text.setPlainText(str(converted))

        # Human-readable previews: try prefer generate_body_text/generate_text
        try:
            a_preview = CardTextGenerator.generate_body_text(action_data, sample=self._sample)
        except Exception:
            try:
                a_preview = CardTextGenerator._format_action(action_data, False, sample=self._sample)
            except Exception:
                a_preview = ""
        self.action_preview.setPlainText(a_preview)

        try:
            c_preview = CardTextGenerator.generate_body_text(converted, sample=self._sample)
        except Exception:
            try:
                c_preview = CardTextGenerator._format_command(converted, False, sample=self._sample)
            except Exception:
                try:
                    c_preview = CardTextGenerator._format_action(converted, False, sample=self._sample)
                except Exception:
                    c_preview = ""
        self.conv_preview.setPlainText(c_preview)

        # Update stat/sample explanation
        try:
            sample_text = self._build_stat_sample_text(action_data, converted, sample=self._sample)
        except Exception:
            sample_text = ""
        self.stat_sample.setPlainText(sample_text)
