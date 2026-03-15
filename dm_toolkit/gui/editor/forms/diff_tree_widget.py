# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt
from typing import Any

class DiffTreeWidget(QWidget):
    """Simple read-only widget representing a diff-tree.

    For testing and headless use it provides `set_diff_tree(tree)` and
    `get_lines()` which returns a list of dotted/bracketed paths for changed leaves.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tree = {}
        self._lines = []
        self._selected: set[str] = set()
        self._selectable = False
        # list widget for selectable lines (falls back to label-less headless usage)
        self.layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        try:
            # prefer enum on newer PyQt
            self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        except Exception:
            try:
                from PyQt6.QtWidgets import QAbstractItemView
                self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
            except Exception:
                # headless/mock widget may not support selection mode API; ignore
                pass
        try:
            self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        except Exception:
            # headless/mock widget may not expose this signal
            pass
        self.layout.addWidget(self.list_widget)

    def set_diff_tree(self, tree: dict[str, Any]):
        self._tree = tree or {}
        self._lines = self._flatten_tree(self._tree)
        # populate list widget
        self.list_widget.clear()
        for line in self._lines:
            item = QListWidgetItem(line)
            # allow toggling via click when selectable (guard for headless mocks)
            try:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable)
            except Exception:
                try:
                    # older Qt enum name
                    item.setFlags(item.flags() | Qt.ItemIsSelectable)
                except Exception:
                    pass
            self.list_widget.addItem(item)
        # keep _selected in sync
        self._selected = set()

    def get_lines(self) -> list[str]:
        return list(self._lines)

    def set_selectable(self, selectable: bool):
        """Enable/disable selection behavior (headless selection still available)."""
        self._selectable = bool(selectable)
        if not self._selectable:
            # clear selection mode
            self.list_widget.clearSelection()

    def select_lines(self, lines: list[str]):
        """Programmatically set selected lines. Lines not present are ignored."""
        self._selected.clear()
        # select items in list_widget
        # If list_widget supports item/count APIs, use them; otherwise update internal set
        try:
            count = self.list_widget.count()
            for i in range(count):
                item = self.list_widget.item(i)
                try:
                    if item.text() in (lines or []):
                        item.setSelected(True)
                        self._selected.add(item.text())
                    else:
                        item.setSelected(False)
                except Exception:
                    if item.text() in (lines or []):
                        self._selected.add(item.text())
                    else:
                        if item.text() in self._selected:
                            self._selected.remove(item.text())
            return
        except Exception:
            # fallback: operate on internal representation
            self._selected.clear()
            for l in (lines or []):
                if l in self._lines:
                    self._selected.add(l)

    def toggle_line_selected(self, line: str):
        """Toggle selection state for a single flattened path string."""
        if line not in self._lines:
            return
        # find item and toggle
        try:
            count = self.list_widget.count()
            for i in range(count):
                item = self.list_widget.item(i)
                if item.text() == line:
                    try:
                        item.setSelected(not item.isSelected())
                    except Exception:
                        if line in self._selected:
                            self._selected.remove(line)
                        else:
                            self._selected.add(line)
                    return
            return
        except Exception:
            # fallback toggle internal set
            if line in self._selected:
                self._selected.remove(line)
            else:
                self._selected.add(line)
            return

    def get_selected_lines(self) -> list[str]:
        return list(self._selected)

    def _on_selection_changed(self):
        # update internal selected set from widget selection
        sel = []
        for item in self.list_widget.selectedItems():
            sel.append(item.text())
        self._selected = set(sel)

    def _flatten_tree(self, tree: dict[str, Any], prefix: str = '') -> list[str]:
        lines: list[str] = []
        if not tree:
            return lines
        for key, val in tree.items():
            # key may be int (list index) or str
            if isinstance(key, int):
                part = f"[{key}]"
            else:
                part = str(key)
            # When key is an int, append as [i] without an extra dot
            if prefix:
                if isinstance(key, int):
                    path = f"{prefix}{part}"
                else:
                    path = f"{prefix}.{part}"
            else:
                path = part
            if val is True:
                lines.append(path)
            elif isinstance(val, dict):
                # recurse
                sub = self._flatten_tree(val, path)
                lines.extend(sub)
            elif val:
                # truthy non-dict: represent as changed
                lines.append(path)
        return lines
