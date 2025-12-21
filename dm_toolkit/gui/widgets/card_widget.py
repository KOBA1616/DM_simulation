# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent


class CardWidget(QFrame):
    clicked = pyqtSignal(int)  # Emits instance_id
    hovered = pyqtSignal(int)  # Emits card_id

    def __init__(self, card_id, card_name, cost, power, civ, tapped=False,
                 instance_id=-1, parent=None, is_face_down=False):
        """
        civ: Can be a single string (e.g. "FIRE")
        or a list of strings (e.g. ["FIRE", "NATURE"]).
        """
        super().__init__(parent)
        self.card_id = card_id
        self.card_name = card_name
        self.cost = cost
        self.power = power

        # Normalize civ to list
        if isinstance(civ, list):
            self.civs = civ
        else:
            self.civs = [civ] if civ else []

        self.tapped = tapped
        self.selected = False  # Selection state
        self.instance_id = instance_id
        self.is_face_down = is_face_down

        self.setFixedSize(100, 140)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)

        # UX Improvement: Cursor Feedback
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # UX Improvement: Accessibility
        self.setAccessibleName(f"Card: {self.card_name}")
        civ_str = "/".join(self.civs)
        self.setAccessibleDescription(
            f"Cost {self.cost}, Power {self.power}, Civilization {civ_str}"
        )

        self.setToolTip(
            f"Name: {self.card_name}\n"
            f"Cost: {self.cost}\n"
            f"Power: {self.power}\n"
            f"Civ: {civ_str}"
        )

        self.init_ui()
        self.update_style()

    def enterEvent(self, event):
        self.hovered.emit(self.card_id)
        super().enterEvent(event)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Header (Cost Circle + Name)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(2)
        header_layout.setContentsMargins(0, 0, 0, 0)

        # Cost Circle
        self.cost_label = QLabel(str(self.cost))
        self.cost_label.setFixedSize(24, 24)
        self.cost_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Hide cost label if face down
        if self.is_face_down:
            self.cost_label.setVisible(False)

        # Style will be set in update_style

        header_layout.addWidget(self.cost_label)

        # Name
        self.name_label = QLabel(self.card_name)
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        font = self.name_label.font()
        font.setBold(True)
        font.setPointSize(8)
        self.name_label.setFont(font)

        header_layout.addWidget(self.name_label)

        layout.addLayout(header_layout)

        layout.addStretch()

        # Footer (Power)
        if self.power > 0:
            self.power_label = QLabel(f"BP:{self.power}")
            self.power_label.setAlignment(
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
            )
            font = self.power_label.font()
            font.setPointSize(7)
            self.power_label.setFont(font)
            layout.addWidget(self.power_label)

    def get_civ_color(self, civ):
        colors_base = {
            "LIGHT": "#DAA520",     # GoldenRod
            "WATER": "#1E90FF",     # DodgerBlue
            "DARKNESS": "#696969",  # DimGray
            "FIRE": "#FF4500",      # OrangeRed
            "NATURE": "#228B22",    # ForestGreen
            "ZERO": "#A9A9A9"       # DarkGray
        }
        return colors_base.get(civ, "#A9A9A9")

    def get_bg_civ_color(self, civ):
        # Lighter colors for card background
        colors_base = {
            "LIGHT": "#FFFACD",     # LemonChiffon
            "WATER": "#E0FFFF",     # LightCyan
            "DARKNESS": "#D3D3D3",  # LightGray
            "FIRE": "#FFE4E1",      # MistyRose
            "NATURE": "#90EE90",    # LightGreen
            "ZERO": "#F5F5F5"       # WhiteSmoke
        }
        return colors_base.get(civ, "#FFFFFF")

    def update_style(self):
        # 1. Update Cost Circle Style (Multicolor Split)
        circle_style = (
            "font-weight: bold; font-size: 10px; color: white; "
            "border: 1px solid black; border-radius: 12px; padding: 0px;"
        )

        if not self.civs:
            circle_bg = "background-color: #A9A9A9;"
        elif len(self.civs) >= 1:
            c = self.get_civ_color(self.civs[0])
            circle_bg = f"background-color: {c};"

        self.cost_label.setStyleSheet(circle_style + circle_bg)

        # 2. Update Card Background Style
        border_color = '#555'
        border_width = '2px'

        if self.tapped:
            border_color = 'red'
            border_width = '3px'

        if self.selected:
            border_color = '#00FF00'  # Bright green for selection
            border_width = '4px'

        if not self.civs:
            bg_style = "background-color: #FFFFFF;"
        elif len(self.civs) == 1:
            c = self.get_bg_civ_color(self.civs[0])
            bg_style = f"background-color: {c};"
        else:
            # Gradient for card background if multi (Linear gradient)
            if len(self.civs) >= 2:
                c1 = self.get_bg_civ_color(self.civs[0])
                c2 = self.get_bg_civ_color(self.civs[1])
                bg_style = (
                    f"background: qlineargradient(spread:pad, "
                    f"x1:0, y1:0, x2:1, y2:1, stop:0 {c1}, stop:1 {c2});"
                )
            else:
                bg_style = "background-color: #E6E6FA;"

        # UX Improvement: Add hover style
        # We need to explicitly define the hover state to change the border color
        # This provides a subtle "lift" or focus effect
        self.setStyleSheet(f"""
            CardWidget {{
                {bg_style}
                border: {border_width} solid {border_color};
                border-radius: 5px;
            }}
            CardWidget:hover {{
                border: {border_width} solid {'#0078d7' if not self.selected else '#32CD32'}; /* Highlight blue or lighter green */
            }}
        """)

    def set_tapped(self, tapped):
        self.tapped = tapped
        self.update_style()

    def set_selected(self, selected):
        self.selected = selected
        self.update_style()

    def mousePressEvent(self, a0):
        if a0 and a0.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.instance_id)
        super().mousePressEvent(a0)
