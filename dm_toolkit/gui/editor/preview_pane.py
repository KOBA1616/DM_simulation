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

        # Card Content Layout
        self.card_layout = QVBoxLayout(self.card_frame)
        self.card_layout.setContentsMargins(15, 15, 15, 15)
        self.card_layout.setSpacing(5)

        # Header (Name, Cost, Civ)
        header_layout = QHBoxLayout()
        self.name_label = QLabel("Card Name")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.name_label.setWordWrap(True)

        self.cost_label = QLabel("5")
        self.cost_label.setStyleSheet("font-weight: bold; font-size: 18px; color: white; background-color: black; border-radius: 15px; padding: 5px;")
        self.cost_label.setFixedSize(30, 30)
        self.cost_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(self.name_label, 1)
        header_layout.addWidget(self.cost_label)
        self.card_layout.addLayout(header_layout)

        # Image Placeholder
        self.image_placeholder = QLabel(tr("[Image]"))
        self.image_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_placeholder.setStyleSheet("border: 1px dashed gray; background-color: #f0f0f0;")
        self.image_placeholder.setFixedHeight(150)
        self.card_layout.addWidget(self.image_placeholder)

        # Type & Race
        self.type_race_label = QLabel("Creature - Race")
        self.type_race_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.type_race_label.setStyleSheet("font-style: italic; font-size: 10px;")
        self.card_layout.addWidget(self.type_race_label)

        # Text Body
        self.text_body = QLabel("Effect text goes here...")
        self.text_body.setWordWrap(True)
        self.text_body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.text_body.setStyleSheet("font-size: 11px;")
        self.card_layout.addWidget(self.text_body, 1)

        # Footer (Power)
        footer_layout = QHBoxLayout()
        self.power_label = QLabel("5000")
        self.power_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        footer_layout.addStretch()
        footer_layout.addWidget(self.power_label)
        self.card_layout.addLayout(footer_layout)

        # Generated Text Preview (Raw)
        layout.addSpacing(20)
        layout.addWidget(QLabel(tr("Generated Text (Raw):")))
        self.raw_text_preview = QTextEdit()
        self.raw_text_preview.setReadOnly(True)
        layout.addWidget(self.raw_text_preview)

    def update_preview(self, item):
        if not item:
            self.clear_preview()
            return

        # Depending on item type, we might want to find the root CARD item
        # But usually we pass the relevant data.
        # If item is CARD, use it. If EFFECT/ACTION, find parent CARD?
        # Actually, the requirement implies showing the card preview.
        # So we should traverse up to find the card data.

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
        self.cost_label.setVisible(True) # Cost always visible?

        races = " / ".join(data.get('races', []))
        type_str = data.get('type', 'CREATURE')
        self.type_race_label.setText(f"{type_str} - {races}" if races else type_str)

        # Effect Text (Simplified for visual, might need parsing or just use generated text)
        # Using generated text for the body is safer
        self.text_body.setText(full_text)

        # Power
        power = data.get('power', 0)
        if power > 0 and 'SPELL' not in type_str:
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
            # Multicolor - simplified to a gradient or mixed color?
            # For now, just use a generic multicolor style or first civ
            bg_color = "qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #FF0000, stop:1 #0000FF)" # Example
            # Let's just pick the first one for simplicity or gold for multi
            bg_color = "#E6E6FA" # Lavender for multi
            border_color = "#4B0082" # Indigo

        self.card_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 4px solid {border_color};
                border-radius: 10px;
            }}
        """)

        # Reset internal labels to transparent so they don't block gradient
        self.name_label.setStyleSheet("background-color: transparent; font-weight: bold; font-size: 14px;")
        self.type_race_label.setStyleSheet("background-color: transparent; font-style: italic; font-size: 10px;")
        self.text_body.setStyleSheet("background-color: transparent; font-size: 11px;")
        self.power_label.setStyleSheet("background-color: transparent; font-weight: bold; font-size: 14px;")

    def clear_preview(self):
        self.name_label.setText("")
        self.cost_label.setText("")
        self.type_race_label.setText("")
        self.text_body.setText("")
        self.power_label.setText("")
        self.raw_text_preview.clear()
        self.card_frame.setStyleSheet("background-color: white; border: 1px solid gray; border-radius: 10px;")
