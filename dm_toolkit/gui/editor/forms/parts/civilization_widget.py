# -*- coding: cp932 -*-
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QButtonGroup
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from dm_toolkit.gui.localization import tr
from dm_toolkit.consts import CIVILIZATIONS

class CivilizationSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Use dynamically loaded civilizations from consts (via dm_ai_module)
        self.civs = CIVILIZATIONS
        self.buttons = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.group = QButtonGroup(self)
        self.group.setExclusive(False)

        # Updated colors for border/text styling
        # Format: (Border/Text Color, Background Hover/Checked Tint)
        civ_styles = {
            "LIGHT":    ("#DAA520", "#FFFACD"),  # Goldenrod / LemonChiffon
            "WATER":    ("#0000FF", "#E0FFFF"),  # Blue / LightCyan
            "DARKNESS": ("#505050", "#D3D3D3"),  # DarkGray / LightGray
            "FIRE":     ("#FF0000", "#FFE4E1"),  # Red / MistyRose
            "NATURE":   ("#008000", "#90EE90"),  # Green / LightGreen
            "ZERO":     ("#808080", "#F5F5F5"),  # Gray / WhiteSmoke
        }

        # Default style for unknown civilizations
        default_style = ("#000000", "#F0F0F0")

        for civ in self.civs:
            btn = QPushButton(tr(civ))
            btn.setCheckable(True)
            btn.setFlat(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

            # Use fixed size or minimum width for better look
            btn.setMinimumWidth(60)

            color_main, color_bg = civ_styles.get(civ, default_style)

            # CSS Styling for colored borders
            style = f"""
                QPushButton {{
                    border: 2px solid {color_main};
                    border-radius: 4px;
                    background-color: #FFFFFF;
                    color: {color_main};
                    font-weight: bold;
                }}
                QPushButton:hover {{
                    background-color: {color_bg};
                }}
                QPushButton:checked {{
                    background-color: {color_main};
                    color: white;
                }}
            """
            btn.setStyleSheet(style)

            layout.addWidget(btn)
            self.group.addButton(btn)
            self.buttons[civ] = btn

            btn.clicked.connect(self.changed.emit)

        layout.addStretch()

    def get_selected_civs(self):
        selected = []
        for civ in self.civs:
            if civ in self.buttons and self.buttons[civ].isChecked():
                selected.append(civ)
        # If empty, logic might interpret as Zero or Colorless elsewhere
        return selected

    def set_selected_civs(self, data):
        # Reset all
        for btn in self.buttons.values():
            btn.setChecked(False)

        if not data:
            return

        if isinstance(data, list):
            for civ in data:
                if civ in self.buttons:
                    self.buttons[civ].setChecked(True)
        elif isinstance(data, str):
            if data in self.buttons:
                self.buttons[data].setChecked(True)

    def blockSignals(self, b):
        for btn in self.buttons.values():
            btn.blockSignals(b)
        return super().blockSignals(b)
