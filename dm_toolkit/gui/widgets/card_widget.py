# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QHBoxLayout, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction, QCursor

from dm_toolkit.gui.styles.civ_colors import CIV_COLORS_FOREGROUND, CIV_COLORS_BACKGROUND
from dm_toolkit.gui.i18n import tr

class CardWidget(QFrame):
    clicked = pyqtSignal(int)  # Emits instance_id
    hovered = pyqtSignal(int)  # Emits card_id
    action_triggered = pyqtSignal(object) # Emits the action object
    double_clicked = pyqtSignal(int)  # Emits instance_id for quick play

    def __init__(self, card_id, card_name, cost, power, civ, tapped=False,
                 instance_id=-1, parent=None, is_face_down=False, legal_actions=None):
        """
        civ: Can be a single string (e.g. "FIRE")
        or a list of strings (e.g. ["FIRE", "NATURE"]).
        legal_actions: List of actions available for this card.
        """
        super().__init__(parent)
        self.card_id = card_id
        self.card_name = card_name
        self.cost = cost
        self.power = power
        self.legal_actions = legal_actions if legal_actions else []

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

        # UX Improvement: Accessibility (localized)
        self.setAccessibleName(f"{tr('CARD')}: {self.card_name}")
        civ_str = "/".join(self.civs)
        self.setAccessibleDescription(
            f"{tr('Cost')} {self.cost}, {tr('Power')} {self.power}, {tr('Civ')} {civ_str}"
        )

        self.setToolTip(
            f"{tr('Name')}: {self.card_name}\n"
            f"{tr('Cost')}: {self.cost}\n"
            f"{tr('Power')}: {self.power}\n"
            f"{tr('Civ')}: {civ_str}"
        )

        self.init_ui()
        self.update_style()

    def update_legal_actions(self, actions):
        self.legal_actions = actions

    def enterEvent(self, event):
        self.hovered.emit(self.card_id)
        super().enterEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to quickly play the default action."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.double_clicked.emit(self.instance_id)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        """Show context menu on right click."""
        if not self.legal_actions:
            return

        menu = QMenu(self)

        # Categorize actions
        # Simplified categorization logic
        added_types = set()

        for action in self.legal_actions:
            action_str = action.to_string() # Fallback description

            # Simple heuristic for display text (localized)
            label = action_str
            if "Play" in action_str:
                label = "カードをプレイ"
            elif "Attack" in action_str:
                if "Player" in action_str:
                    label = "プレイヤーを攻撃"
                elif "Creature" in action_str:
                    label = "クリーチャーを攻撃"
                else:
                    label = "攻撃"
            elif "Mana" in action_str:
                label = tr("MANA_CHARGE")
            elif "Use Ability" in action_str:
                label = "能力を使用"

            # De-duplicate identical labels if multiple similar actions exist (e.g. attack different shields)
            # For simplicity, if we have multiple Attack Player (different shields), we might want to just show one "Attack Player"
            # and let the engine resolve/ask target, BUT the engine usually generates distinct actions.
            # For now, list them all but try to be descriptive.

            act = QAction(label, self)
            # Use a closure to capture the specific action
            act.triggered.connect(lambda checked, a=action: self.action_triggered.emit(a))
            menu.addAction(act)

        menu.exec(event.globalPos())

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
        return CIV_COLORS_FOREGROUND.get(civ, "#A9A9A9")

    def get_bg_civ_color(self, civ):
        # Lighter colors for card background
        return CIV_COLORS_BACKGROUND.get(civ, "#FFFFFF")

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
        self.setStyleSheet(f"""
            CardWidget {{
                {bg_style}
                border: {border_width} solid {border_color};
                border-radius: 5px;
            }}
            CardWidget:hover {{
                border: {border_width} solid {'#0078d7' if not self.selected else '#32CD32'};
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
