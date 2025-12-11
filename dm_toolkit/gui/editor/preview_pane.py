from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QTextEdit, QFrame, QGridLayout,
    QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

class CardPreviewWidget(QWidget):
    """
    A widget that displays a visual preview of the card and its generated text.
    Acts as the third pane in the Card Editor.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_data = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Title
        title = QLabel(tr("Card Preview"))
        font = title.font()
        font.setBold(True)
        font.setPointSize(12)
        title.setFont(font)
        layout.addWidget(title)

        # Visual Card Representation (Mock)
        self.card_frame = QFrame()
        self.card_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.card_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.card_frame.setLineWidth(2)
        self.card_frame.setFixedSize(300, 420) # Approx card ratio
        self.card_frame.setStyleSheet("background-color: white; border-radius: 10px;")

        # Center the card frame
        h_layout = QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(self.card_frame)
        h_layout.addStretch()
        layout.addLayout(h_layout)

        # Card Content Layout: Use Grid for precise positioning
        self.card_layout = QGridLayout(self.card_frame)
        self.card_layout.setContentsMargins(10, 10, 10, 10)
        self.card_layout.setSpacing(5)

        # Row 0: Header (Cost - Name)
        # Requirement: "Cost circle to top-left", "Name"
        self.cost_label = QLabel("5")
        self.cost_label.setStyleSheet("font-weight: bold; font-size: 18px; color: white; background-color: black; border-radius: 15px; padding: 2px;")
        self.cost_label.setFixedSize(30, 30)
        self.cost_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.name_label = QLabel("Card Name")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px; background-color: transparent;")
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.card_layout.addWidget(self.cost_label, 0, 0)
        self.card_layout.addWidget(self.name_label, 0, 1, 1, 2) # Span 2 columns

        # Row 1: Race (Requirement: "Display Race under the Name")
        self.race_label = QLabel("Race")
        self.race_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.race_label.setStyleSheet("font-style: italic; font-size: 10px; background-color: transparent;")
        self.card_layout.addWidget(self.race_label, 1, 1, 1, 2)

        # Row 2: Type (Optional placement, putting below race or merged)
        # Requirement: "Type display ... change to Keyword/Effect area"
        # Let's keep type distinct but small
        self.type_label = QLabel("[Creature]")
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.type_label.setStyleSheet("font-weight: bold; font-size: 10px; background-color: transparent;")
        self.card_layout.addWidget(self.type_label, 1, 0) # Below cost?

        # Row 3: Image removed (Requirement: "Image space removal")
        # Instead, expand text body.

        # Row 3-4: Text Body (Keywords + Effects)
        # Requirement: "Generated Text display ... change to Keyword/Ability display area"
        self.text_body = QLabel("")
        self.text_body.setWordWrap(True)
        self.text_body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.text_body.setStyleSheet("font-size: 11px; background-color: rgba(255, 255, 255, 0.5); border-radius: 5px; padding: 5px;")
        self.card_layout.addWidget(self.text_body, 2, 0, 1, 3) # Span all columns, take remaining space

        # Row 5: Footer (Power)
        # Requirement: "Power numerical value to bottom-left"
        self.power_label = QLabel("5000")
        self.power_label.setStyleSheet("font-weight: bold; font-size: 16px; color: black; background-color: transparent;")
        self.power_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)

        self.card_layout.addWidget(self.power_label, 3, 0)

        # Set row stretches to push footer down and text body expand
        self.card_layout.setRowStretch(2, 1) # Text body takes available space

        # Generated Text Preview (Raw) - keeping as debug/copy source
        layout.addSpacing(10)
        layout.addWidget(QLabel(tr("Generated Text (Source):")))
        self.raw_text_preview = QTextEdit()
        self.raw_text_preview.setReadOnly(True)
        self.raw_text_preview.setFixedHeight(100)
        layout.addWidget(self.raw_text_preview)

    def update_preview(self, item):
        if not item:
            self.clear_preview()
            return

        model = item.model()
        parent = item
        card_item = None

        while parent:
            type_ = parent.data(Qt.ItemDataRole.UserRole + 1)
            if type_ == "CARD":
                card_item = parent
                break
            parent = parent.parent()

        if not card_item:
            self.clear_preview()
            return

        data = card_item.data(Qt.ItemDataRole.UserRole + 2)
        if not data:
            return

        self.current_data = data
        self.render_card(data)

    def render_card(self, data):
        # Generate Text
        full_text = CardTextGenerator.generate_text(data)
        self.raw_text_preview.setText(full_text)

        # Update Visuals
        self.name_label.setText(data.get('name', '???'))
        self.cost_label.setText(str(data.get('cost', 0)))
        self.cost_label.setVisible(True)

        races = " / ".join(data.get('races', []))
        self.race_label.setText(races if races else "")

        type_str = CardTextGenerator.TYPE_MAP.get(data.get('type', 'CREATURE'), data.get('type', ''))
        self.type_label.setText(f"[{type_str}]")

        # Effect Text (Simplified for visual, might need parsing or just use generated text)
        # Use full generated text which includes keywords now
        # But we might want to strip the header from full_text since we display header separately
        # The generator puts header in lines[0] etc.
        # Let's strip the header lines for the body display

        lines = full_text.split('\n')
        body_lines = []
        skip_mode = True
        for line in lines:
            if line.startswith("■"):
                skip_mode = False

            if not skip_mode:
                body_lines.append(line)
            elif "--------------------" in line: # Divider
                skip_mode = False

        # If no divider/keywords found, just show all (fallback)
        if not body_lines and len(lines) > 3:
             body_lines = lines[3:]

        self.text_body.setText("\n".join(body_lines))

        # Power
        power = data.get('power', 0)
        if power > 0 and 'SPELL' not in data.get('type', ''):
            self.power_label.setText(str(power))
            self.power_label.setVisible(True)
        else:
            self.power_label.setVisible(False)

        # Civilization Colors
        civs = data.get('civilizations', [])
        if not civs and 'civilization' in data:
            civs = [data['civilization']]

        self.apply_civ_style(civs)

    def apply_civ_style(self, civs):
        # Basic mapping
        colors = {
            "LIGHT": "#FFFACD",     # LemonChiffon
            "WATER": "#E0FFFF",     # LightCyan
            "DARKNESS": "#D3D3D3",  # LightGray
            "FIRE": "#FFE4E1",      # MistyRose
            "NATURE": "#90EE90",    # LightGreen
            "ZERO": "#F5F5F5"       # WhiteSmoke
        }

        border_colors = {
            "LIGHT": "#DAA520",
            "WATER": "#0000FF",
            "DARKNESS": "#505050",
            "FIRE": "#FF0000",
            "NATURE": "#008000",
            "ZERO": "#808080"
        }

        if not civs:
            bg_color = "#FFFFFF"
            border_color = "#000000"
        elif len(civs) == 1:
            c = civs[0]
            bg_color = colors.get(c, "#FFFFFF")
            border_color = border_colors.get(c, "#000000")
        else:
            # Gradient for multicolor (2 colors)
            if len(civs) >= 2:
                c1 = colors.get(civs[0], "#FFFFFF")
                c2 = colors.get(civs[1], "#FFFFFF")
                bg_color = f"qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 {c1}, stop:1 {c2})"
                border_color = "#4B0082" # Indigo
            else:
                bg_color = "#E6E6FA"
                border_color = "#4B0082"

        self.card_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color if 'gradient' not in bg_color else 'transparent'};
                background: {bg_color};
                border: 4px solid {border_color};
                border-radius: 10px;
            }}
        """)

        # Note: 'background' property works for gradients in stylesheet

    def clear_preview(self):
        self.name_label.setText("")
        self.cost_label.setText("")
        self.race_label.setText("")
        self.type_label.setText("")
        self.text_body.setText("")
        self.power_label.setText("")
        self.raw_text_preview.clear()
        self.card_frame.setStyleSheet("background-color: white; border: 1px solid gray; border-radius: 10px;")
