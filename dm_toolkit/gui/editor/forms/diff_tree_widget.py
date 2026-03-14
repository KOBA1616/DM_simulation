# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
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
        # simple label for UI; optional
        self.layout = QVBoxLayout(self)
        self.label = QLabel("")
        self.layout.addWidget(self.label)

    def set_diff_tree(self, tree: dict[str, Any]):
        self._tree = tree or {}
        self._lines = self._flatten_tree(self._tree)
        self.label.setText('\n'.join(self._lines))

    def get_lines(self) -> list[str]:
        return list(self._lines)

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
