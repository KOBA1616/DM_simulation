from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette, QMouseEvent

class CardWidget(QFrame):
    clicked = pyqtSignal(int) # Emits instance_id
    hovered = pyqtSignal(int) # Emits card_id

    def __init__(self, card_id, card_name, cost, power, civ, tapped=False, instance_id=-1, parent=None):
        """
        civ: Can be a single string (e.g. "FIRE") or a list of strings (e.g. ["FIRE", "NATURE"]).
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
        self.instance_id = instance_id
        
        self.setFixedSize(100, 140)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        self.setLineWidth(2)

        civ_str = "/".join(self.civs)
        self.setToolTip(f"Name: {self.card_name}\nCost: {self.cost}\nPower: {self.power}\nCiv: {civ_str}")
        
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
        # Style will be set in update_style
        
        header_layout.addWidget(self.cost_label)

        # Name
        self.name_label = QLabel(self.card_name)
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
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
            self.power_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
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
        circle_style = "font-weight: bold; font-size: 10px; color: white; border: 1px solid black; border-radius: 12px; padding: 0px;"

        if not self.civs:
             circle_bg = "background-color: #A9A9A9;"
        elif len(self.civs) == 1:
            c = self.get_civ_color(self.civs[0])
            circle_bg = f"background-color: {c};"
        else:
            # Conical gradient for split circle
            stops = []
            # Ensure unique civs to prevent weird single-color multi-segments
            unique_civs = []
            seen = set()
            for c in self.civs:
                if c not in seen:
                    unique_civs.append(c)
                    seen.add(c)

            n = len(unique_civs)
            for i, civ in enumerate(unique_civs):
                c = self.get_civ_color(civ)
                # Hard stops for segments
                start = i / n
                end = (i + 1) / n

                # Format to 3 decimal places to avoid scientific notation and ensure hard stops
                # stop:0.500 #Red, stop:0.500 #Blue creates a hard edge in Qt
                stops.append(f"stop:{start:.3f} {c}")
                stops.append(f"stop:{end:.3f} {c}")

            grad_str = ", ".join(stops)
            # angle:90 starts at 12 o'clock. For 2 colors: 0-0.5 (Left), 0.5-1.0 (Right).
            circle_bg = f"background: qconicalgradient(cx:0.5, cy:0.5, angle:90, {grad_str});"

        self.cost_label.setStyleSheet(circle_style + circle_bg)

        # 2. Update Card Background Style
        border_color = '#555' if not self.tapped else 'red'
        border_width = '2px' if not self.tapped else '3px'

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
                bg_style = f"background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 {c1}, stop:1 {c2});"
            else:
                 bg_style = "background-color: #E6E6FA;"

        self.setStyleSheet(f"""
            CardWidget {{
                {bg_style}
                border: {border_width} solid {border_color};
                border-radius: 5px;
            }}
        """)

    def set_tapped(self, tapped):
        self.tapped = tapped
        self.update_style()

    def mousePressEvent(self, a0: QMouseEvent | None):
        if a0 and a0.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.instance_id)
        super().mousePressEvent(a0)
