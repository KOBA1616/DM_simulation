from typing import List, Dict, Any
import os
import json

from dm_toolkit.gui.editor.validators_shared import generate_missing_ids

# Allow tests and headless environments to force a pure-Python fallback by
# setting environment variable `DM_EDITOR_HEADLESS=1` before importing this module.
_FORCE_HEADLESS = os.environ.get('DM_EDITOR_HEADLESS') == '1'

_HAS_QT = True
try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem,
        QPlainTextEdit, QLabel, QSpinBox, QFormLayout
    )
    from PyQt6.QtCore import Qt
    from dm_toolkit.gui.editor.forms.parts.draggable_list import DraggableListWidget
    from dm_toolkit.gui.editor.forms.signal_utils import safe_connect
except Exception:
    _HAS_QT = False

# Honor forced headless mode even if PyQt is importable
if _FORCE_HEADLESS:
    _HAS_QT = False


class _CostReductionEditorFallback:
    """Pure-Python fallback with the same public API used in tests.

    This allows importing the module and running non-GUI unit tests in CI or
    other headless environments where Qt platform plugins are unavailable.
    """

    def __init__(self, parent=None):
        self._items: List[Dict[str, Any]] = []
        self._selected_index = 0

    def get_value(self) -> List[Dict[str, Any]]:
        return [dict(i) for i in self._items]

    def set_value(self, val: List[Dict[str, Any]]):
        if not isinstance(val, list):
            self._items = []
            return
        # ensure ids for each entry
        for entry in val:
            generate_missing_ids([entry])
        self._items = [dict(e) for e in val]

    def _on_add(self):
        entry = {"type": "PASSIVE", "amount": 1}
        generate_missing_ids([entry])
        self._items.append(entry)

    def _on_remove(self):
        if self._items:
            self._items.pop()

    # Selection and input-assist helpers for headless tests
    def set_selected_index(self, idx: int):
        if not self._items:
            self._selected_index = 0
            return
        self._selected_index = max(0, min(idx, len(self._items) - 1))

    def update_selected_fields(self, amount: int | None = None, min_mana_cost: int | None = None, unit_cost: int | None = None, max_units: int | None = None):
        if not self._items:
            return
        entry = self._items[self._selected_index]
        if amount is not None:
            entry['amount'] = int(amount)
        if min_mana_cost is not None:
            entry['min_mana_cost'] = int(min_mana_cost)
        if unit_cost is not None:
            entry['unit_cost'] = int(unit_cost)
        if max_units is not None:
            entry['max_units'] = int(max_units)
        # ensure ids still present
        generate_missing_ids([entry])
        self._items[self._selected_index] = entry

    def suggest_input_assist(self, context: Dict[str, Any] | None = None) -> Dict[str, int]:
        """Return suggested values for `unit_cost`, `max_units`, `min_mana_cost` for headless fallback.

        Uses the same heuristic as the Qt-backed implementation.
        """
        if not self._items:
            return {"unit_cost": 1, "max_units": 1, "min_mana_cost": 0}
        entry = self._items[self._selected_index]
        try:
            unit_cost = int(entry.get('unit_cost', 1))
        except Exception:
            unit_cost = 1
        try:
            max_units = int(entry.get('max_units', entry.get('amount', 1)))
        except Exception:
            max_units = int(entry.get('amount', 1) or 1)
        try:
            min_mana = int(entry.get('min_mana_cost', 0))
        except Exception:
            min_mana = 0
        if min_mana == 0:
            min_mana = (unit_cost * max_units) // 2
        return {"unit_cost": unit_cost, "max_units": max_units, "min_mana_cost": min_mana}

    def compute_effective_cost(self, units: int | None = None) -> int:
        """Compute an editor-preview effective cost for the selected entry.

        Heuristic used in editor preview:
        - applied_units = min(units, max_units) if max_units > 0 else units
        - base_cost = unit_cost * applied_units
        - effective_cost = max(base_cost, min_mana_cost)
        """
        if not self._items:
            return 0
        entry = self._items[self._selected_index]
        u = units if units is not None else int(entry.get('amount', 1))
        try:
            unit_cost = int(entry.get('unit_cost', 0))
        except Exception:
            unit_cost = 0
        try:
            max_units = int(entry.get('max_units')) if entry.get('max_units') is not None else u
        except Exception:
            max_units = u
        try:
            min_mana = int(entry.get('min_mana_cost', 0))
        except Exception:
            min_mana = 0
        applied_units = u
        if max_units and max_units > 0:
            applied_units = min(u, max_units)
        base_cost = unit_cost * applied_units
        return max(base_cost, min_mana)

    def get_preview_text(self, units: int | None = None) -> str:
        c = self.compute_effective_cost(units)
        u = units if units is not None else 1
        return f"Preview ({u} unit{'s' if u != 1 else ''}): {c}"


if not _HAS_QT:
    CostReductionEditor = _CostReductionEditorFallback
else:

    class CostReductionEditor(QWidget):
        """Qt-backed editor for `cost_reductions`.

        Exposes `get_value()` and `set_value(list)` so it can be integrated with `CardEditForm`.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self._init_ui()

        def _init_ui(self):
            layout = QHBoxLayout(self)

            left = QVBoxLayout()
            self.list_widget = DraggableListWidget()
            left.addWidget(QLabel("Entries"))
            left.addWidget(self.list_widget)

            btns = QHBoxLayout()
            self.add_btn = QPushButton("Add")
            self.remove_btn = QPushButton("Remove")
            self.up_btn = QPushButton("Up")
            self.down_btn = QPushButton("Down")
            btns.addWidget(self.add_btn)
            btns.addWidget(self.remove_btn)
            btns.addWidget(self.up_btn)
            btns.addWidget(self.down_btn)
            left.addLayout(btns)

            right = QVBoxLayout()
            right.addWidget(QLabel("Selected Entry (JSON)"))
            self.json_edit = QPlainTextEdit()
            right.addWidget(self.json_edit)

            # Input-assist controls (amount, min_mana_cost, unit_cost, max_units)
            form = QFormLayout()
            self.amount_spin = QSpinBox()
            self.amount_spin.setMinimum(0)
            self.amount_spin.setMaximum(999)
            form.addRow(QLabel("amount"), self.amount_spin)

            self.min_mana_spin = QSpinBox()
            self.min_mana_spin.setMinimum(0)
            self.min_mana_spin.setMaximum(99)
            form.addRow(QLabel("min_mana_cost"), self.min_mana_spin)

            self.unit_cost_spin = QSpinBox()
            self.unit_cost_spin.setMinimum(0)
            self.unit_cost_spin.setMaximum(99)
            form.addRow(QLabel("unit_cost"), self.unit_cost_spin)

            self.max_units_spin = QSpinBox()
            self.max_units_spin.setMinimum(0)
            self.max_units_spin.setMaximum(999)
            form.addRow(QLabel("max_units"), self.max_units_spin)

            right.addLayout(form)

            # Preview display
            right.addWidget(QLabel("Preview"))
            self.preview_label = QLabel("")
            right.addWidget(self.preview_label)

            layout.addLayout(left, 1)
            layout.addLayout(right, 2)

            safe_connect(self.add_btn, 'clicked', self._on_add)
            safe_connect(self.remove_btn, 'clicked', self._on_remove)
            safe_connect(self.up_btn, 'clicked', lambda: self._move_selected(-1))
            safe_connect(self.down_btn, 'clicked', lambda: self._move_selected(1))
            safe_connect(self.list_widget, 'currentItemChanged', self._on_selection_changed)
            safe_connect(self.json_edit, 'textChanged', self._on_json_changed)

            # Bind input controls to update_selected_fields
            safe_connect(self.amount_spin, 'valueChanged', lambda v: self.update_selected_fields(amount=v))
            safe_connect(self.min_mana_spin, 'valueChanged', lambda v: self.update_selected_fields(min_mana_cost=v))
            safe_connect(self.unit_cost_spin, 'valueChanged', lambda v: self.update_selected_fields(unit_cost=v))
            safe_connect(self.max_units_spin, 'valueChanged', lambda v: self.update_selected_fields(max_units=v))

            self._suppress_json_update = False

        def _on_add(self):
            entry = {"type": "PASSIVE", "amount": 1}
            generate_missing_ids([entry])
            item = QListWidgetItem(entry.get('id', ''))
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.list_widget.addItem(item)
            self.list_widget.setCurrentItem(item)

        def _on_remove(self):
            item = self.list_widget.currentItem()
            if item:
                row = self.list_widget.row(item)
                self.list_widget.takeItem(row)
                self.json_edit.setPlainText('')

        def _move_selected(self, delta: int):
            item = self.list_widget.currentItem()
            if not item:
                return
            row = self.list_widget.row(item)
            new_row = row + delta
            if new_row < 0 or new_row >= self.list_widget.count():
                return
            it = self.list_widget.takeItem(row)
            self.list_widget.insertItem(new_row, it)
            self.list_widget.setCurrentItem(it)

        def _on_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
            if current:
                entry = current.data(Qt.ItemDataRole.UserRole) or {}
                self._suppress_json_update = True
                try:
                    self.json_edit.setPlainText(json.dumps(entry, ensure_ascii=False, indent=2))
                finally:
                    self._suppress_json_update = False
            else:
                self.json_edit.setPlainText('')

            # refresh input controls values
            if current:
                entry = current.data(Qt.ItemDataRole.UserRole) or {}
                try:
                    self.amount_spin.setValue(int(entry.get('amount', 0)))
                except Exception:
                    pass
                try:
                    self.min_mana_spin.setValue(int(entry.get('min_mana_cost', 0)))
                except Exception:
                    pass
                try:
                    self.unit_cost_spin.setValue(int(entry.get('unit_cost', 0)))
                except Exception:
                    pass
                try:
                    self.max_units_spin.setValue(int(entry.get('max_units', 0)))
                except Exception:
                    pass

            # update preview when selection changes
            self._update_preview_display()

        def _on_json_changed(self):
            if self._suppress_json_update:
                return
            item = self.list_widget.currentItem()
            if not item:
                return
            txt = self.json_edit.toPlainText()
            try:
                parsed = json.loads(txt) if txt.strip() else {}
            except Exception:
                # Ignore parse errors until the user fixes them
                return
            # Ensure id presence
            generate_missing_ids([parsed])
            item.setData(Qt.ItemDataRole.UserRole, parsed)
            item.setText(parsed.get('id', ''))

            # update preview when JSON edited
            self._update_preview_display()

        # Integration API for CardEditForm
        def get_value(self) -> List[Dict[str, Any]]:
            result = []
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                data = item.data(Qt.ItemDataRole.UserRole) or {}
                result.append(data)
            return result

        def set_value(self, val: List[Dict[str, Any]]):
            self.list_widget.clear()
            if not isinstance(val, list):
                return
            for entry in val:
                # Ensure each has an id for display
                generate_missing_ids([entry])
                item = QListWidgetItem(entry.get('id', ''))
                item.setData(Qt.ItemDataRole.UserRole, entry)
                self.list_widget.addItem(item)
            # select first
            if self.list_widget.count() > 0:
                self.list_widget.setCurrentRow(0)

        # Selection and input-assist helpers for tests and UI callers
        def set_selected_index(self, idx: int):
            if self.list_widget.count() == 0:
                return
            idx = max(0, min(idx, self.list_widget.count() - 1))
            self.list_widget.setCurrentRow(idx)

        def update_selected_fields(self, amount: int | None = None, min_mana_cost: int | None = None, unit_cost: int | None = None, max_units: int | None = None):
            item = self.list_widget.currentItem()
            if not item:
                return
            entry = item.data(Qt.ItemDataRole.UserRole) or {}
            if amount is not None:
                entry['amount'] = int(amount)
            if min_mana_cost is not None:
                entry['min_mana_cost'] = int(min_mana_cost)
            if unit_cost is not None:
                entry['unit_cost'] = int(unit_cost)
            if max_units is not None:
                entry['max_units'] = int(max_units)
            generate_missing_ids([entry])
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setText(entry.get('id', ''))

            # refresh preview label
            self._update_preview_display()

            # refresh input-assist hints (if UI wants them)
            try:
                hints = self.suggest_input_assist()
                # if UI hook exists, set properties; keep optional to avoid tight coupling
                if hasattr(self, 'apply_input_assist_hints'):
                    self.apply_input_assist_hints(hints)
            except Exception:
                pass

        def _update_preview_display(self, units: int | None = None):
            try:
                text = self.get_preview_text(units=units)
            except Exception:
                text = "Preview: N/A"
            self.preview_label.setText(text)

        # Input-assist suggestion API for UI callers
    def suggest_input_assist(self, context: Dict[str, Any] | None = None) -> Dict[str, int]:
        """Return suggested values for `unit_cost`, `max_units`, `min_mana_cost`.

        Heuristic (editor-side):
        - `unit_cost`: existing or 1
        - `max_units`: existing `max_units` or `amount` or 1
        - `min_mana_cost`: existing or floor(unit_cost * max_units / 2)
        """
        item = self.list_widget.currentItem()
        if not item:
            return {"unit_cost": 1, "max_units": 1, "min_mana_cost": 0}
        entry = item.data(Qt.ItemDataRole.UserRole) or {}
        try:
            unit_cost = int(entry.get('unit_cost', 1))
        except Exception:
            unit_cost = 1
        try:
            max_units = int(entry.get('max_units', entry.get('amount', 1)))
        except Exception:
            max_units = int(entry.get('amount', 1) or 1)
        try:
            min_mana = int(entry.get('min_mana_cost', 0))
        except Exception:
            min_mana = 0
        if min_mana == 0:
            min_mana = (unit_cost * max_units) // 2
        return {"unit_cost": unit_cost, "max_units": max_units, "min_mana_cost": min_mana}

            # Preview helpers for UI and tests
        def compute_effective_cost(self, units: int | None = None) -> int:
            """Compute an editor-preview effective cost for the currently selected entry.

            Uses the same heuristic as the headless fallback.
            """
            item = self.list_widget.currentItem()
            if not item:
                return 0
            entry = item.data(Qt.ItemDataRole.UserRole) or {}
            u = units if units is not None else int(entry.get('amount', 1))
            try:
                unit_cost = int(entry.get('unit_cost', 0))
            except Exception:
                unit_cost = 0
            try:
                max_units = int(entry.get('max_units')) if entry.get('max_units') is not None else u
            except Exception:
                max_units = u
            try:
                min_mana = int(entry.get('min_mana_cost', 0))
            except Exception:
                min_mana = 0
            applied_units = u
            if max_units and max_units > 0:
                applied_units = min(u, max_units)
            base_cost = unit_cost * applied_units
            return max(base_cost, min_mana)

        def get_preview_text(self, units: int | None = None) -> str:
            c = self.compute_effective_cost(units)
            u = units if units is not None else 1
            return f"Preview ({u} unit{'s' if u != 1 else ''}): {c}"
