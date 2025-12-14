from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QTextEdit, QFrame, QGridLayout,
    QHBoxLayout, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from dm_toolkit.gui.localization import tr
from dm_toolkit.gui.editor.text_generator import CardTextGenerator

class CardPreviewWidget(QWidget):
    """
    A widget that displays a visual preview of the card and its generated text.
    Acts as the third pane in the Card Editor.
    Updated to support Twinpact (Split View) with requested UI refinements.
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

        # Visual Card Representation Frame
        self.card_frame = QFrame()
        self.card_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.card_frame.setFrameShadow(QFrame.Shadow.Raised)
        self.card_frame.setLineWidth(2)
        self.card_frame.setFixedSize(300, 420) # Approx card ratio
        # Default style, will be overridden by apply_civ_style
        self.card_frame.setStyleSheet("background-color: white; border-radius: 10px; border: 1px solid black;")

        # Center the card frame
        h_layout = QHBoxLayout()
        h_layout.addStretch()
        h_layout.addWidget(self.card_frame)
        h_layout.addStretch()
        layout.addLayout(h_layout)

        # Main Layout inside the card (Switchable between Standard and Twinpact)
        self.frame_layout = QVBoxLayout(self.card_frame)
        self.frame_layout.setContentsMargins(0,0,0,0)
        self.frame_layout.setSpacing(0)

        # Standard View Container
        self.standard_widget = QWidget()
        self.standard_layout = QGridLayout(self.standard_widget)
        self.setup_standard_layout(self.standard_layout)

        # Twinpact View Container
        self.twinpact_widget = QWidget()
        self.twinpact_layout = QVBoxLayout(self.twinpact_widget)
        self.setup_twinpact_layout(self.twinpact_layout)

        self.frame_layout.addWidget(self.standard_widget)
        self.frame_layout.addWidget(self.twinpact_widget)

        # Hide both initially
        self.standard_widget.hide()
        self.twinpact_widget.hide()

        # Generated Text Preview (Raw)
        layout.addSpacing(10)
        layout.addWidget(QLabel(tr("Generated Text (Source):")))
        self.raw_text_preview = QTextEdit()
        self.raw_text_preview.setReadOnly(True)
        self.raw_text_preview.setFixedHeight(100)
        layout.addWidget(self.raw_text_preview)

    def setup_standard_layout(self, layout):
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # Cost (Top Left)
        self.cost_label = QLabel("5")
        self.cost_label.setFixedSize(30, 30)
        self.cost_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.cost_label, 0, 0)

        # Name
        self.name_label = QLabel("Card Name")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px; background-color: transparent;")
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.name_label, 0, 1, 1, 2)

        # Race
        self.race_label = QLabel("Race")
        self.race_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.race_label.setStyleSheet("font-style: italic; font-size: 10px; background-color: transparent;")
        layout.addWidget(self.race_label, 1, 1, 1, 2)

        # Type
        self.type_label = QLabel("[Creature]")
        self.type_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.type_label.setStyleSheet("font-weight: bold; font-size: 10px; background-color: transparent;")
        layout.addWidget(self.type_label, 1, 0)

        # Text Body
        self.text_body = QLabel("")
        self.text_body.setWordWrap(True)
        self.text_body.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        # Requirement: Thin black border for text box
        self.text_body.setStyleSheet("font-size: 11px; background-color: rgba(255, 255, 255, 0.5); border: 1px solid black; border-radius: 5px; padding: 5px;")
        layout.addWidget(self.text_body, 2, 0, 1, 3)

        # Power (Bottom Left)
        self.power_label = QLabel("5000")
        self.power_label.setStyleSheet("font-weight: bold; font-size: 16px; color: black; background-color: transparent;")
        self.power_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.power_label, 3, 0)

        layout.setRowStretch(2, 1)

    def setup_twinpact_layout(self, layout):
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Upper Half (Creature)
        self.tp_upper_frame = QFrame()
        self.tp_upper_frame.setStyleSheet("background-color: transparent; border: 1px solid gray; border-radius: 5px;")
        upper_layout = QGridLayout(self.tp_upper_frame)
        upper_layout.setContentsMargins(5,5,5,5)

        self.tp_cost_label = QLabel("5")
        self.tp_cost_label.setFixedSize(24, 24)
        self.tp_cost_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        upper_layout.addWidget(self.tp_cost_label, 0, 0)

        self.tp_name_label = QLabel("Creature Name")
        self.tp_name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        upper_layout.addWidget(self.tp_name_label, 0, 1)

        self.tp_race_label = QLabel("Race")
        self.tp_race_label.setStyleSheet("font-style: italic; font-size: 9px;")
        upper_layout.addWidget(self.tp_race_label, 1, 1)

        self.tp_body_label = QLabel("Creature Text")
        self.tp_body_label.setWordWrap(True)
        self.tp_body_label.setStyleSheet("font-size: 10px; background-color: rgba(255,255,255,0.4);")
        self.tp_body_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        upper_layout.addWidget(self.tp_body_label, 2, 0, 1, 2)
        upper_layout.setRowStretch(2, 1)

        # REMOVED Power from Upper Frame to move to Card Bottom Left
        # upper_layout.addWidget(self.tp_power_label, 3, 0)

        layout.addWidget(self.tp_upper_frame)

        # Lower Half (Spell)
        self.tp_lower_frame = QFrame()
        self.tp_lower_frame.setStyleSheet("background-color: transparent; border: 1px solid gray; border-radius: 5px;")
        lower_layout = QGridLayout(self.tp_lower_frame)
        lower_layout.setContentsMargins(5,5,5,5)

        # Spell Cost at Top Right (Requirement)
        self.tp_spell_cost_label = QLabel("3")
        self.tp_spell_cost_label.setFixedSize(24, 24)
        self.tp_spell_cost_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lower_layout.addWidget(self.tp_spell_cost_label, 0, 2, Qt.AlignmentFlag.AlignRight) # Column 2

        self.tp_spell_name_label = QLabel("Spell Name")
        self.tp_spell_name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        lower_layout.addWidget(self.tp_spell_name_label, 0, 0, 1, 2)

        self.tp_spell_body_label = QLabel("Spell Text")
        self.tp_spell_body_label.setWordWrap(True)
        self.tp_spell_body_label.setStyleSheet("font-size: 10px; background-color: rgba(255,255,255,0.4);")
        self.tp_spell_body_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        lower_layout.addWidget(self.tp_spell_body_label, 1, 0, 1, 3)
        lower_layout.setRowStretch(1, 1)

        # Power moved to Lower Frame (Card Bottom Left)
        self.tp_power_label = QLabel("5000")
        self.tp_power_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.tp_power_label.setAlignment(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft)
        # Adding at row 2, col 0 (below text, left side)
        lower_layout.addWidget(self.tp_power_label, 2, 0)

        layout.addWidget(self.tp_lower_frame)
        layout.setStretch(0, 5) # Creature roughly 50-60%?
        layout.setStretch(1, 4) # Spell

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

    def clear_preview(self):
        self.standard_widget.hide()
        self.twinpact_widget.hide()
        self.raw_text_preview.clear()

    def render_card(self, data):
        full_text = CardTextGenerator.generate_text(data)
        self.raw_text_preview.setText(full_text)

        # Check Twinpact
        is_twinpact = 'spell_side' in data and data['spell_side'] is not None

        civs = data.get('civilizations', [])
        if not civs and 'civilization' in data:
            civs = [data['civilization']]

        if is_twinpact:
            self.standard_widget.hide()
            self.twinpact_widget.show()
            self.render_twinpact(data, civs)
        else:
            self.twinpact_widget.hide()
            self.standard_widget.show()
            self.render_standard(data, civs)

        self.apply_civ_style(civs)

    def render_standard(self, data, civs):
        self.name_label.setText(data.get('name', '???'))
        self.cost_label.setText(str(data.get('cost', 0)))

        # Apply cost circle color
        self.apply_cost_circle_style(self.cost_label, civs)

        races = " / ".join(data.get('races', []))
        self.race_label.setText(races if races else "")

        type_str = CardTextGenerator.TYPE_MAP.get(data.get('type', 'CREATURE'), data.get('type', ''))
        self.type_label.setText(f"[{type_str}]")

        body_text = self.extract_body_text(CardTextGenerator.generate_text(data))
        self.text_body.setText(body_text)

        power = data.get('power', 0)
        if power > 0 and 'SPELL' not in data.get('type', ''):
            self.power_label.setText(str(power))
            self.power_label.setVisible(True)
        else:
            self.power_label.setVisible(False)

    def render_twinpact(self, data, civs):
        # Creature Side (Main)
        self.tp_name_label.setText(data.get('name', '???'))
        self.tp_cost_label.setText(str(data.get('cost', 0)))

        # Apply cost circle color for creature side
        self.apply_cost_circle_style(self.tp_cost_label, civs)

        races = " / ".join(data.get('races', []))
        self.tp_race_label.setText(races if races else "")
        self.tp_power_label.setText(str(data.get('power', 0)))

        # Hack: Generate text for ONLY the creature part
        creature_data = data.copy()
        if 'spell_side' in creature_data:
            del creature_data['spell_side']
        creature_text = CardTextGenerator.generate_text(creature_data)
        self.tp_body_label.setText(self.extract_body_text(creature_text))

        # Spell Side
        spell_data = data.get('spell_side', {})
        self.tp_spell_name_label.setText(spell_data.get('name', 'Spell'))
        self.tp_spell_cost_label.setText(str(spell_data.get('cost', 0)))

        # Apply cost circle color for spell side (Spell usually matches card civs)
        self.apply_cost_circle_style(self.tp_spell_cost_label, civs)

        # For spell text, we can use the generator on the spell data object
        if 'type' not in spell_data:
            spell_data['type'] = 'SPELL'
        spell_text = CardTextGenerator.generate_text(spell_data)
        self.tp_spell_body_label.setText(self.extract_body_text(spell_text))

    def extract_body_text(self, full_text):
        lines = full_text.split('\n')
        body_lines = []
        skip_mode = True
        for line in lines:
            if line.startswith("■") or line.startswith("S・トリガー") or line.startswith("G・ストライク"):
                skip_mode = False

            # Additional heuristic: If line contains standard keywords
            if any(k in line for k in ["ブロッカー", "W・ブレイカー", "T・ブレイカー", "スピードアタッカー"]):
                skip_mode = False

            if not skip_mode:
                body_lines.append(line)
            elif "--------------------" in line:
                skip_mode = False

        if not body_lines and len(lines) > 2:
            return "\n".join(lines[2:]) # Skip header lines if heuristics fail

        return "\n".join(body_lines)

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

    def apply_cost_circle_style(self, label, civs):
        # Requirement: "Mana cost circle is composed of civilization colors (divided if multi)."
        # "White text with black border" (Simulating black border via text shadow if possible, else just bold white)

        # NOTE: Qt QLabel doesn't support -webkit-text-stroke.
        # Using a simplistic approach: Bold White Text on colored background.
        # Adding a border to the circle itself.

        style = "font-weight: bold; font-size: 16px; color: white; border: 2px solid black; border-radius: 12px; padding: 0px;"

        if not civs:
            bg_style = "background-color: #A9A9A9;"
        elif len(civs) == 1:
            c = self.get_civ_color(civs[0])
            bg_style = f"background-color: {c};"
        else:
            # Conical gradient for split circle (Cake-like)
            stops = []
            # Ensure unique civs
            unique_civs = []
            seen = set()
            for c in civs:
                if c not in seen:
                    unique_civs.append(c)
                    seen.add(c)

            n = len(unique_civs)
            for i, civ in enumerate(unique_civs):
                c = self.get_civ_color(civ)
                # Hard stops for segments
                start = i / n
                end = (i + 1) / n
                stops.append(f"stop:{start:.3f} {c}")
                stops.append(f"stop:{end:.3f} {c}")

            # Conical gradient starting at 135 degrees (Top-Left diagonal)
            grad_str = ", ".join(stops)
            bg_style = f"background: qconicalgradient(cx:0.5, cy:0.5, angle:135, {grad_str});"

        label.setStyleSheet(style + bg_style)

    def apply_civ_style(self, civs):
        # Updated mapping (Darker Gradients requested)
        colors_base = {
            "LIGHT": "#FFFACD",     # LemonChiffon
            "WATER": "#E0FFFF",     # LightCyan
            "DARKNESS": "#D3D3D3",  # LightGray
            "FIRE": "#FFE4E1",      # MistyRose
            "NATURE": "#90EE90",    # LightGreen
            "ZERO": "#F5F5F5"       # WhiteSmoke
        }

        # Requirement: "All borders should be thin black lines"
        border_color = "#000000"

        if not civs:
            bg_style = "background-color: #FFFFFF;"
        elif len(civs) == 1:
            c = civs[0]
            c1 = colors_base.get(c, "#FFFFFF")
            # Solid color as requested
            bg_style = f"background-color: {c1};"
        else:
            if len(civs) >= 2:
                c1 = colors_base.get(civs[0], "#FFFFFF")
                c2 = colors_base.get(civs[1], "#FFFFFF")
                bg_style = f"background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 {c1}, stop:1 {c2});"
            else:
                bg_style = "background-color: #E6E6FA;"

        self.card_frame.setStyleSheet(f"""
            QFrame {{
                {bg_style}
                border: 1px solid {border_color};
                border-radius: 10px;
            }}
        """)
