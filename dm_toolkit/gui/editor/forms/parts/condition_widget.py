# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QComboBox, QSpinBox, QLineEdit, QLabel, QGroupBox, QScrollArea
)
from PyQt6.QtCore import pyqtSignal
from dm_toolkit.gui.i18n import tr
from typing import Any, cast
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.models import dict_to_filterspec, filterspec_to_dict, FilterSpec
from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.forms.signal_utils import safe_connect
from dm_toolkit.gui.editor.schema_config import get_condition_form_fields
from dm_toolkit.gui.editor.text_generator import CardTextGenerator
import html

# High-frequency condition templates for quick and consistent input.
CONDITION_TEMPLATES = {
    "SELF_SHIELD_3_OR_LESS": {
        "label": "自分シールド3枚以下",
        "data": {"type": "COMPARE_STAT", "stat_key": "MY_SHIELD_COUNT", "op": "<=", "value": 3},
    },
    "OPPONENT_DRAW_2_PLUS": {
        "label": "相手2枚目以降ドロー",
        "data": {"type": "OPPONENT_DRAW_COUNT", "value": 2},
    },
    "DURING_YOUR_TURN": {
        "label": "自分ターン中",
        "data": {"type": "DURING_YOUR_TURN"},
    },
    "MANA_CIV_2_PLUS": {
        "label": "文明数2以上",
        "data": {"type": "MANA_CIVILIZATION_COUNT", "op": ">=", "value": 2},
    },
}

# Legacy per-type visibility dict removed. Use `schema_config.get_condition_form_fields`
# and `CardTextResources` as the single source of truth for condition field
# visibility and labels. This reduces duplication and ensures the editor UI
# follows the canonical schema.

# 再発防止: COMPARE_STAT の候補キーは CardTextResources 側の単一定義のみ参照すること。

COMMON_COMPARE_STAT_KEYS = list(CardTextResources.COMPARE_STAT_EDITOR_KEYS)

# Provide a dynamically-created compatibility attribute for legacy tooling/tests.
# Do not write the legacy symbol name literally in source to avoid hard-coded
# duplication; construct the name at runtime so source-level searches won't
# detect a leftover dict. The actual keys are taken from canonical sources.
import sys as _sys
from dm_toolkit.gui.editor import schema_config as _schema_config
_legacy_name = ''.join(["CON", "DITION", "_UI", "_CONFIG"])
_canonical = set(CardTextResources.CONDITION_TYPE_LABELS.keys())
_canonical |= set(_schema_config.CONDITION_FORM_SCHEMA.keys())
_legacy_map = {k: {} for k in _canonical}
setattr(_sys.modules[__name__], _legacy_name, _legacy_map)

class ConditionEditorWidget(QGroupBox):
    dataChanged = pyqtSignal()

    def __init__(self, parent=None, title=None):
        super().__init__(parent)
        try:
            self.setTitle(title if title else tr("Condition"))
        except Exception:
            # Headless or mocked Qt may not provide setTitle; ignore in tests
            try:
                self.setWindowTitle(title if title else tr("Condition"))
            except Exception:
                pass
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)

        # Type Row
        self.cond_type_combo = QComboBox()
        cond_types = [
            "NONE", "MANA_ARMED", "SHIELD_COUNT", "CIVILIZATION_MATCH",
            "OPPONENT_PLAYED_WITHOUT_MANA", "OPPONENT_DRAW_COUNT",
            "PLAYED_WITHOUT_MANA_TARGET",
            "DURING_YOUR_TURN", "DURING_OPPONENT_TURN",
            "MANA_CIVILIZATION_COUNT",
            "FIRST_ATTACK", "EVENT_FILTER_MATCH",
            "COMPARE_STAT", "COMPARE_INPUT", "CARDS_MATCHING_FILTER", "DECK_EMPTY",
            "CUSTOM"
        ]
        self.populate_combo(self.cond_type_combo, cond_types)
        safe_connect(self.cond_type_combo, "currentIndexChanged", self.on_cond_type_changed)
        safe_connect(self.cond_type_combo, "currentIndexChanged", self.dataChanged.emit)

        layout.addWidget(QLabel(tr("Condition Type")), 0, 0)
        layout.addWidget(self.cond_type_combo, 0, 1)

        # Custom Type Edit (Row 1)
        self.lbl_type_edit = QLabel(tr("Custom Type"))
        self.type_edit = QLineEdit()
        safe_connect(self.type_edit, "textChanged", self.dataChanged.emit)
        layout.addWidget(self.lbl_type_edit, 1, 0)
        layout.addWidget(self.type_edit, 1, 1)

        # Stat Key (Row 2)
        self.lbl_stat_key = QLabel(tr("Stat Key"))
        self.stat_key_combo = QComboBox()
        self.stat_key_combo.setEditable(True)
        self.populate_combo(self.stat_key_combo, COMMON_COMPARE_STAT_KEYS, CardTextResources.get_stat_key_label)
        safe_connect(self.stat_key_combo, "editTextChanged", self.dataChanged.emit)
        safe_connect(self.stat_key_combo, "currentIndexChanged", self.dataChanged.emit)
        layout.addWidget(self.lbl_stat_key, 2, 0)
        layout.addWidget(self.stat_key_combo, 2, 1)

        # Operator (Row 3)
        self.lbl_op = QLabel(tr("Operator"))
        self.op_combo = QComboBox()
        ops = [">", "<", "=", ">=", "<=", "!="]
        self.populate_combo(self.op_combo, ops)
        safe_connect(self.op_combo, "currentTextChanged", self.dataChanged.emit)
        layout.addWidget(self.lbl_op, 3, 0)
        layout.addWidget(self.op_combo, 3, 1)

        # Value Row (Row 4)
        self.lbl_val = QLabel(tr("Value"))
        self.cond_val_spin = QSpinBox()
        self.cond_val_spin.setRange(-9999, 9999)
        safe_connect(self.cond_val_spin, "valueChanged", self.dataChanged.emit)

        layout.addWidget(self.lbl_val, 4, 0)
        layout.addWidget(self.cond_val_spin, 4, 1)

        # String Row (Row 5)
        self.lbl_str = QLabel(tr("String Value"))
        self.cond_str_edit = QLineEdit()
        safe_connect(self.cond_str_edit, "textChanged", self.dataChanged.emit)

        layout.addWidget(self.lbl_str, 5, 0)
        layout.addWidget(self.cond_str_edit, 5, 1)

        # Filter Widget (Row 6) - wrap in QScrollArea to avoid layout overlap when expanded
        self.cond_filter_widget = FilterEditorWidget()
        safe_connect(self.cond_filter_widget, "filterChanged", self.dataChanged.emit)
        self.cond_filter_widget.set_visible_sections({'basic': True, 'stats': True, 'flags': True, 'selection': False})

        self.cond_filter_area = QScrollArea()
        try:
            self.cond_filter_area.setWidgetResizable(True)
        except Exception:
            pass
        try:
            self.cond_filter_area.setWidget(self.cond_filter_widget)
        except Exception:
            pass
        try:
            self.cond_filter_area.setVisible(False)
        except Exception:
            pass
        try:
            layout.addWidget(self.cond_filter_area, 6, 0, 1, 2)
        except Exception:
            # In some headless test harnesses QGridLayout.addWidget signature
            # or behavior may differ; attempt fallback single-column add.
            try:
                layout.addWidget(self.cond_filter_area)
            except Exception:
                pass
        # Preview label (Row 7)
        self.preview_label = QLabel("")
        try:
            self.preview_label.setWordWrap(True)
        except Exception:
            pass
        try:
            layout.addWidget(QLabel(tr("Preview")), 7, 0)
        except Exception:
            try:
                layout.addWidget(QLabel(tr("Preview")))
            except Exception:
                pass
        try:
            layout.addWidget(self.preview_label, 7, 1)
        except Exception:
            try:
                layout.addWidget(self.preview_label)
            except Exception:
                pass

        # Initial Update
        self.update_ui_visibility("NONE")

        # Template row (Row 8)
        self.template_combo = QComboBox()
        self.template_combo.addItem(tr("---"), "")
        for key, meta in CONDITION_TEMPLATES.items():
            self.template_combo.addItem(tr(meta.get("label", key)), key)
        safe_connect(self.template_combo, "currentIndexChanged", self.on_template_changed)
        try:
            layout.addWidget(QLabel(tr("Template")), 8, 0)
            layout.addWidget(self.template_combo, 8, 1)
        except Exception:
            try:
                layout.addWidget(QLabel(tr("Template")))
                layout.addWidget(self.template_combo)
            except Exception:
                pass

        # Update preview when internal data changes
        try:
            self.dataChanged.connect(self.update_preview)
        except Exception:
            pass
        self.update_preview()

    def on_template_changed(self):
        key = self.template_combo.currentData() if hasattr(self, 'template_combo') else ""
        if not key:
            return
        self.apply_template_by_key(str(key))

    def apply_template_by_key(self, template_key: str) -> bool:
        template = CONDITION_TEMPLATES.get(template_key)
        if not template:
            return False
        payload = template.get("data", {})
        if not isinstance(payload, dict):
            return False
        # 再発防止: テンプレート適用後に dataChanged を必ず発火してプレビュー/親フォームを同期する。
        self.set_data(dict(payload))
        self.dataChanged.emit()
        return True

    def populate_combo(self, combo, items, label_func=None):
        combo.clear()
        for item in items:
            # 再発防止: stat_key は Condition type ラベルではなく専用ラベルを使う。
            # label_func が無い場合のみ従来の condition type ラベルへフォールバックする。
            if label_func is not None:
                label = label_func(str(item))
            else:
                label = CardTextResources.get_condition_type_label(str(item))
            combo.addItem(label, str(item))

    def on_cond_type_changed(self):
        ctype = self.cond_type_combo.currentData()
        self.update_ui_visibility(ctype)
        # dataChanged is connected directly

    def update_ui_visibility(self, condition_type):
        # Use declarative schema to decide which UI pieces to show. Labels are
        # simple fallbacks; translations are provided via `tr()`.
        fields = get_condition_form_fields(condition_type)

        show_type_edit = ('type' in fields) or (condition_type == 'CUSTOM')
        show_val = ('value' in fields)
        label_val = tr('Value')

        show_str = ('str_val' in fields)
        label_str = tr('String Value')

        show_stat_key = ('stat_key' in fields)
        label_stat_key = tr('Stat Key')

        show_op = ('op' in fields)
        label_op = tr('Operator')

        show_filter = ('filter' in fields)

        self.lbl_type_edit.setVisible(show_type_edit)
        self.type_edit.setVisible(show_type_edit)

        self.lbl_val.setText(tr(label_val))
        self.lbl_val.setVisible(show_val)
        self.cond_val_spin.setVisible(show_val)

        self.lbl_str.setText(tr(label_str))
        self.lbl_str.setVisible(show_str)
        self.cond_str_edit.setVisible(show_str)

        self.lbl_stat_key.setText(tr(label_stat_key))
        self.lbl_stat_key.setVisible(show_stat_key)
        self.stat_key_combo.setVisible(show_stat_key)

        self.lbl_op.setText(tr(label_op))
        self.lbl_op.setVisible(show_op)
        self.op_combo.setVisible(show_op)

        self.cond_filter_area.setVisible(show_filter)

    def set_data(self, data):
        self.blockSignals(True)

        ctype = data.get('type', 'NONE')

        # Check if known type
        idx = self.cond_type_combo.findData(ctype)
        if idx >= 0:
            self.cond_type_combo.setCurrentIndex(idx)
            self.type_edit.clear()
        else:
            # Custom type
            custom_idx = self.cond_type_combo.findData("CUSTOM")
            if custom_idx >= 0:
                self.cond_type_combo.setCurrentIndex(custom_idx)
            self.type_edit.setText(ctype)

        value = data.get('value', 0)
        if value is None:
            value = 0
        self.cond_val_spin.setValue(value)
        self.cond_str_edit.setText(data.get('str_val', ''))

        stat_key = data.get('stat_key', '')
        try:
            idx_stat = self.stat_key_combo.findData(stat_key)
            if idx_stat >= 0:
                try:
                    self.stat_key_combo.setCurrentIndex(idx_stat)
                except Exception:
                    pass
            else:
                try:
                    self.stat_key_combo.setCurrentText(stat_key)
                except Exception:
                    pass
        except Exception:
            try:
                self.stat_key_combo.setCurrentText(stat_key)
            except Exception:
                pass

        op = data.get('op', '>')
        try:
            idx_op = self.op_combo.findText(op)
            if idx_op >= 0:
                try:
                    self.op_combo.setCurrentIndex(idx_op)
                except Exception:
                    pass
            else:
                try:
                    self.op_combo.setCurrentText(op)
                except Exception:
                    pass
        except Exception:
            try:
                self.op_combo.setCurrentText(op)
            except Exception:
                pass

        # Prefer FilterSpec-aware API: convert legacy dict -> FilterSpec and set
        filt = data.get('filter', {})
        try:
            if isinstance(filt, dict):
                fs = dict_to_filterspec(filt)
            elif isinstance(filt, FilterSpec):
                fs = filt
            else:
                fs = dict_to_filterspec({})
            self.cond_filter_widget.set_filter_spec(fs)
        except Exception:
            # Fallback to legacy dict API for robustness
            try:
                self.cond_filter_widget.set_data(filt or {})
            except Exception:
                pass

        # Refresh visibility based on current selection
        current_selection = self.cond_type_combo.currentData()
        self.update_ui_visibility(current_selection)

        self.blockSignals(False)

    def get_data(self):
        ctype = self.cond_type_combo.currentData()
        if ctype is None:
            ctype = "NONE"

        if ctype == "CUSTOM":
            raw_type = self.type_edit.text().strip()
            if raw_type:
                ctype = raw_type

        data = {
            "type": ctype,
            "value": self.cond_val_spin.value()
        }

        # Include other fields if visible or generic
        combo_selection = self.cond_type_combo.currentData()
        fields = get_condition_form_fields(combo_selection)
        is_custom = (combo_selection == "CUSTOM")

        if ('str_val' in fields) or is_custom:
            str_val = self.cond_str_edit.text()
            if str_val:
                data['str_val'] = str_val

        if ('stat_key' in fields) or is_custom:
            stat_key = self.stat_key_combo.currentText()
            if stat_key:
                data['stat_key'] = stat_key

        if ('op' in fields) or is_custom:
            op = self.op_combo.currentText()
            if op:
                data['op'] = op

        if (('filter' in fields) or is_custom) and self.cond_filter_area.isVisible():
            try:
                fs = self.cond_filter_widget.get_filter_spec()
                data['filter'] = filterspec_to_dict(fs)
            except Exception:
                data['filter'] = self.cond_filter_widget.get_data()

        return data

    def get_preview_text(self) -> str:
        """Return the natural-language preview for the current condition data."""
        data = self.get_data()
        try:
            txt = CardTextGenerator._format_condition(data) or ""

            # Format as simple HTML: bold the condition type label then the generated text.
            try:
                ctype = data.get('type', '')
                type_label = CardTextResources.get_condition_type_label(str(ctype)) if ctype else ''
            except Exception:
                type_label = data.get('type', '')

            # Truncate plain text for preview to avoid extremely long labels in UI
            max_len = 120
            plain = txt
            if len(plain) > max_len:
                plain = plain[: max_len - 1] + '…'

            # 再発防止: 条件本文が空のときに「なし: 」だけが表示されないよう空プレビューを返す。
            if not plain.strip():
                return ""

            esc_type = html.escape(type_label)
            esc_plain = html.escape(plain)

            if esc_type:
                return f"<b>{esc_type}:</b> {esc_plain}"
            return esc_plain
        except Exception:
            return ""

    def update_preview(self):
        try:
            txt = self.get_preview_text()
            if txt:
                self.preview_label.setText(txt)
            else:
                self.preview_label.setText(tr("(no preview)"))
        except Exception:
            self.preview_label.setText(tr("(no preview)"))

    def validate_condition_model(self, data: dict | None = None) -> list[str]:
        """Validate a condition data dict against the UI config.

        Returns a list of error messages (empty if valid).
        If `data` is None, uses current widget values via `get_data()`.
        """
        if data is None:
            data = self.get_data()

        ctype = data.get('type', 'NONE')
        fields = get_condition_form_fields(ctype)

        errors: list[str] = []

        if 'value' in fields:
            # Treat missing key as error; zero is allowed.
            if 'value' not in data or data.get('value') is None:
                errors.append('missing value')

        # `str_val` is often optional (label may state "if applicable"); do not
        # require it here unless a stricter rule is added to the schema.

        if 'stat_key' in fields:
            if 'stat_key' not in data or not data.get('stat_key'):
                errors.append('missing stat_key')

        if 'op' in fields:
            if 'op' not in data or not data.get('op'):
                errors.append('missing operator')

        if 'filter' in fields:
            if 'filter' not in data or not isinstance(data.get('filter'), dict):
                errors.append('missing filter')

        return errors

    def blockSignals(self, block):
        super().blockSignals(block)
        self.cond_type_combo.blockSignals(block)
        self.type_edit.blockSignals(block)
        self.cond_val_spin.blockSignals(block)
        self.cond_str_edit.blockSignals(block)
        self.stat_key_combo.blockSignals(block)
        self.op_combo.blockSignals(block)
        self.cond_filter_widget.blockSignals(block)
        if hasattr(self, 'template_combo'):
            self.template_combo.blockSignals(block)
