from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QButtonGroup
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from dm_toolkit.gui.localization import tr

class CivilizationSelector(QWidget):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.civs = ["LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"]
        self.buttons = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.group = QButtonGroup(self)
        self.group.setExclusive(False)

        civ_colors = {
            "LIGHT": "#DAA520",     # Goldenrod
            "WATER": "#0000FF",     # Blue
            "DARKNESS": "#505050",  # Dark Gray
            "FIRE": "#FF0000",      # Red
            "NATURE": "#008000",    # Green
            "ZERO": "#808080"       # Gray
        }

        # Japanese labels if needed, but usually we map internal keys to translated labels
        # For buttons, we might use short text or icons. Let's use 1-2 letters or translated name.
        # Assuming tr() handles "LIGHT" -> "光" etc.

        for civ in self.civs:
            btn = QPushButton(tr(civ))
            btn.setCheckable(True)
            btn.setFlat(False)
            btn.setToolTip(tr(civ))

            # Set style for colored background when checked
            color = civ_colors.get(civ, "#FFFFFF")
            # We want the color to show when checked.
            # When unchecked, standard button style.
            # When checked, solid color with white text (usually).

            # Note: Qt styling can be tricky.
            # Simple approach: setStyleSheet
            style = f"""
                QPushButton:checked {{
                    background-color: {color};
                    color: white;
                    border: 2px solid #333;
                }}
                QPushButton {{
                    font-weight: bold;
                }}
            """
            btn.setStyleSheet(style)

            layout.addWidget(btn)
            self.group.addButton(btn)
            self.buttons[civ] = btn

            btn.clicked.connect(self.changed.emit)

    def get_selected_civs(self):
        selected = []
        for civ in self.civs:
            if self.buttons[civ].isChecked():
                selected.append(civ)
        return selected

    def set_selected_civs(self, data):
        # Reset all
        for btn in self.buttons.values():
            btn.setChecked(False) # setChecked doesn't trigger clicked usually, but we should be careful with signals

        if not data:
            return

        # Handle list of strings
        if isinstance(data, list):
            for civ in data:
                if civ in self.buttons:
                    self.buttons[civ].setChecked(True)
        # Handle single string (legacy)
        elif isinstance(data, str):
            if data in self.buttons:
                self.buttons[data].setChecked(True)

    def blockSignals(self, b):
        for btn in self.buttons.values():
            btn.blockSignals(b)
        return super().blockSignals(b)
