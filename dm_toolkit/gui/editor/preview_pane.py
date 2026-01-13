# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QTextEdit, QFrame, QGridLayout,
    QHBoxLayout, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QFont, QColor, QPainter, QPen
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.text_generator import CardTextGenerator
from dm_toolkit.gui.editor import normalize
from dm_toolkit.gui.styles.civ_colors import CIV_COLORS_FOREGROUND, CIV_COLORS_BACKGROUND

class ManaCostLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.civs = []
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Base text style. Background is handled by paintEvent.
        self.setStyleSheet("font-weight: bold; font-size: 16px; color: white; background-color: transparent; padding: 0px;")

    def set_civs(self, civs):
        self.civs = civs
        self.update() # Trigger repaint

    def get_civ_color(self, civ):
        return QColor(CIV_COLORS_FOREGROUND.get(civ, "#A9A9A9"))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate square drawing area centered in the widget
        rect = self.rect()
        side = min(rect.width(), rect.height())

        # Calculate offsets to center the square
        x_offset = (rect.width() - side) // 2
        y_offset = (rect.height() - side) // 2

        # Define the drawing rect with a margin for the border
        # Border width is 2, so we need 1px margin on each side to keep it inside
        margin = 1
        draw_size = side - 2 * margin

        if draw_size <= 0:
            return

        draw_rect = QRect(x_offset + margin, y_offset + margin, draw_size, draw_size)

        if not self.civs:
             painter.setBrush(QColor("#A9A9A9"))
             painter.setPen(Qt.PenStyle.NoPen)
             painter.drawEllipse(draw_rect)
        elif len(self.civs) == 1:
             painter.setBrush(self.get_civ_color(self.civs[0]))
             painter.setPen(Qt.PenStyle.NoPen)
             painter.drawEllipse(draw_rect)
        else:
            # Draw distinct sectors (Pies)
            n = len(self.civs)
            total_span = 360 * 16
            start_angle_base = 90 * 16 # 12 o'clock

            for i, civ in enumerate(self.civs):
                painter.setBrush(self.get_civ_color(civ))
                painter.setPen(Qt.PenStyle.NoPen)

                # Calculate angles precisely to avoid gaps
                angle_start = start_angle_base + (total_span * i) // n
                angle_next = start_angle_base + (total_span * (i + 1)) // n
                span = angle_next - angle_start

                painter.drawPie(draw_rect, angle_start, span)

        # Draw Border
        pen = QPen(Qt.GlobalColor.black)
        pen.setWidth(2)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawEllipse(draw_rect)

        # Draw Text (Number)
        font = self.font()
        font.setBold(True)
        # Calculate font size relative to the actual drawing area
        # Using approx 60% of the inner diameter for better visibility
        font_size = max(8, int(draw_size * 0.6))
        font.setPixelSize(font_size)
        painter.setFont(font)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(draw_rect, Qt.AlignmentFlag.AlignCenter, self.text())

        painter.end()

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

        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
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

        # Generated Text Preview (Raw) disabled
        self.raw_text_preview = QTextEdit()
        self.raw_text_preview.setReadOnly(True)
        self.raw_text_preview.setFixedHeight(100)
        self.raw_text_preview.setVisible(False)

    def setup_standard_layout(self, layout):
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        # Cost (Top Left)
        self.cost_label = ManaCostLabel("5")
        self.cost_label.setFixedSize(30, 30)
        layout.addWidget(self.cost_label, 0, 0)

        # Name
        self.name_label = QLabel(tr("Card Name"))
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px; background-color: transparent;")
        self.name_label.setWordWrap(True)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.name_label, 0, 1, 1, 2)

        # Race
        self.race_label = QLabel(tr("Race"))
        self.race_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.race_label.setStyleSheet("font-style: italic; font-size: 10px; background-color: transparent;")
        layout.addWidget(self.race_label, 1, 1, 1, 2)

        # Type
        self.type_label = QLabel(tr("[Creature]"))
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
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(1)

        # Upper Half (Creature)
        self.tp_upper_frame = QFrame()
        self.tp_upper_frame.setStyleSheet("background-color: transparent; border: 1px solid gray; border-radius: 5px;")
        upper_layout = QGridLayout(self.tp_upper_frame)
        upper_layout.setContentsMargins(5,5,5,5)

        self.tp_cost_label = ManaCostLabel("5")
        self.tp_cost_label.setFixedSize(24, 24)
        upper_layout.addWidget(self.tp_cost_label, 0, 0)

        self.tp_name_label = QLabel(tr("Creature Name"))
        self.tp_name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        upper_layout.addWidget(self.tp_name_label, 0, 1)

        self.tp_race_label = QLabel(tr("Race"))
        self.tp_race_label.setStyleSheet("font-style: italic; font-size: 9px;")
        upper_layout.addWidget(self.tp_race_label, 1, 1)

        self.tp_body_label = QLabel(tr("Creature Text"))
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
        self.tp_spell_cost_label = ManaCostLabel("3")
        self.tp_spell_cost_label.setFixedSize(24, 24)
        lower_layout.addWidget(self.tp_spell_cost_label, 0, 2, Qt.AlignmentFlag.AlignRight) # Column 2

        self.tp_spell_name_label = QLabel(tr("Spell Name"))
        self.tp_spell_name_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        lower_layout.addWidget(self.tp_spell_name_label, 0, 0, 1, 2)

        self.tp_spell_body_label = QLabel(tr("Spell Text"))
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

    def clear_preview(self):
        self.standard_widget.hide()
        self.twinpact_widget.hide()
        self.raw_text_preview.clear()

    def render_card(self, data):
        # When raw text preview is hidden, skip expensive text generation.
        if self.raw_text_preview.isVisible():
            full_text = CardTextGenerator.generate_text(data)
            # Generate canonical summaries for preview (CIR) to help detect action/command mismatches
            cir_lines = []
            effects = data.get('effects', []) or data.get('triggers', []) or []
            for ei, eff in enumerate(effects, start=1):
                commands = eff.get('commands', [])
                if commands:
                    for ci, cmd in enumerate(commands, start=1):
                        cir = normalize.canonicalize(cmd)
                        kind = cir.get('kind')
                        ctype = cir.get('type') or ''
                        opts = cir.get('options') or []
                        branches = cir.get('branches') or {}
                        opt_info = f" options={len(opts)}" if opts else ""
                        branch_info = ''
                        if branches:
                            t = len(branches.get('if_true', []))
                            f = len(branches.get('if_false', []))
                            branch_info = f" branches=({t}/{f})"
                        cir_lines.append(f"{tr('Effect')}[{ei}] {tr('Command')}[{ci}]: {kind}/{ctype}{opt_info}{branch_info}")
                # Legacy actions processing removed (Migration Phase 4.3)

            summary = "\n".join(cir_lines)
            if summary:
                synth = self._build_natural_summaries(data)
                if synth:
                    self.raw_text_preview.setText(synth + "\n\n" + full_text + "\n\n" + tr("CIR Summary:") + "\n" + summary)
                else:
                    self.raw_text_preview.setText(full_text + "\n\n" + tr("CIR Summary:") + "\n" + summary)
            else:
                synth = self._build_natural_summaries(data)
                if synth:
                    self.raw_text_preview.setText(synth + "\n\n" + full_text)
                else:
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

    def _should_show_power(self, data):
        """
        Consolidated logic to determine if power should be displayed.
        """
        try:
            power = int(data.get('power', 0))
        except (ValueError, TypeError):
            return False

        if power <= 0:
            return False

        raw_type = data.get('type', '')
        if isinstance(raw_type, list):
            t = " ".join([str(x) for x in raw_type]).upper()
        else:
            t = str(raw_type).upper()

        # Logic: Only show power if the type explicitly indicates it is a Creature.
        # This aligns with standard Duel Masters rules where only Creatures (and variants) have Power.
        # It prevents power from showing on Spells, Cross Gears, Castles, etc. even if power > 0 is set in data.
        is_creature = 'CREATURE' in t

        return is_creature

    def render_standard(self, data, civs):
        self.name_label.setText(data.get('name', '???'))
        self.cost_label.setText(str(data.get('cost', 0)))

        # Apply cost circle color
        self.apply_cost_circle_style(self.cost_label, civs)

        races = " / ".join(data.get('races', []))
        self.race_label.setText(races if races else "")

        type_str = CardTextGenerator.TYPE_MAP.get(data.get('type', 'CREATURE'), data.get('type', ''))
        self.type_label.setText(f"[{type_str}]")

        # Use new structure (generator returns a string)
        body_text = CardTextGenerator.generate_body_text_lines(data)
        self.text_body.setText(body_text)

        if self._should_show_power(data):
            self.power_label.setText(str(data.get('power', 0)))
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

        if self._should_show_power(data):
            self.tp_power_label.setText(str(data.get('power', 0)))
            self.tp_power_label.setVisible(True)
        else:
            self.tp_power_label.setVisible(False)

        # Generate text for ONLY the creature part (generator returns a string)
        creature_text = CardTextGenerator.generate_body_text_lines(data, include_twinpact=False)
        self.tp_body_label.setText(creature_text)

        # Spell Side
        spell_data = data.get('spell_side', {})
        self.tp_spell_name_label.setText(spell_data.get('name', 'Spell'))
        self.tp_spell_cost_label.setText(str(spell_data.get('cost', 0)))

        # Apply cost circle color for spell side (Spell usually matches card civs)
        self.apply_cost_circle_style(self.tp_spell_cost_label, civs)

        # For spell text, we can use the generator on the spell data object
        if 'type' not in spell_data:
            spell_data['type'] = 'SPELL'

        spell_text = CardTextGenerator.generate_body_text_lines(spell_data)
        self.tp_spell_body_label.setText(spell_text)

    def _build_natural_summaries(self, data):
        """Scan effects/actions to create compact natural-language summary sentences
        for common patterns (e.g., draw N then move N to deck bottom on enter).
        """
        lines = []
        effects = data.get('effects', []) or data.get('triggers', []) or []
        for eff in effects:
            trig = eff.get('trigger', 'NONE')
            trigger_text = CardTextGenerator.trigger_to_japanese(trig)

            # Commands-First Policy
            raw_ops = eff.get('commands', [])
            if not raw_ops:
                raw_ops = eff.get('actions', []) or []

            # Look for draw then deck-bottom pair
            if len(raw_ops) >= 2:
                a0 = raw_ops[0]
                a1 = raw_ops[1]
                # detect draw (TRANSITION DECK->HAND or DRAW_CARD)
                def is_draw(a):
                    t = a.get('type', '')
                    if t == 'DRAW_CARD':
                        return True
                    if t == 'TRANSITION':
                        f = (a.get('from_zone') or a.get('fromZone') or '').upper()
                        to = (a.get('to_zone') or a.get('toZone') or '').upper()
                        if (f == '' or 'DECK' in f) and 'HAND' in to:
                            return True
                    return False

                def is_deck_bottom_move(a):
                    dest = (a.get('destination_zone') or a.get('to_zone') or a.get('toZone') or '').upper()
                    if 'DECK_BOTTOM' in dest or 'DECKBOTTOM' in dest:
                        return True
                    t = (a.get('type') or '').upper()
                    if 'DECK_BOTTOM' in t:
                        return True
                    return False

                if is_draw(a0) and is_deck_bottom_move(a1):
                    # Build natural sentence
                    # If draw amount explicit, include it; otherwise describe condition if present
                    amt = a0.get('amount') or a0.get('value1') or None
                    if not amt:
                        # Try to infer from target_filter.count or condition referencing mana civs
                        tf = a0.get('target_filter') or {}
                        amt = tf.get('count') or None

                    if amt:
                        draw_part = f"{trigger_text}、山札からカードを{amt}枚引く。"
                    else:
                        # fallback phrasing for dynamic count (e.g., based on マナ文明の数)
                        draw_part = f"{trigger_text}、マナゾーンの文明と数と同じ枚数までカードを引く。"

                    tail = "その後、引いた枚数と同じ枚数を山札の下に置く。"
                    lines.append(draw_part + tail)

        return "\n".join(lines)

    # Deprecated / Fallback
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

    def apply_cost_circle_style(self, label, civs):
        # Delegate to ManaCostLabel if applicable
        if isinstance(label, ManaCostLabel):
            label.set_civs(civs)
            return

        # Fallback implementation removed to ensure consistency.
        # All cost labels should be ManaCostLabel instances.

    def apply_civ_style(self, civs):
        # Requirement: "All borders should be thin black lines"
        border_color = "#000000"

        if not civs:
            bg_style = "background-color: #FFFFFF;"
        elif len(civs) == 1:
            c = civs[0]
            c1 = CIV_COLORS_BACKGROUND.get(c, "#FFFFFF")
            # Solid color as requested
            bg_style = f"background-color: {c1};"
        else:
            if len(civs) >= 2:
                c1 = CIV_COLORS_BACKGROUND.get(civs[0], "#FFFFFF")
                c2 = CIV_COLORS_BACKGROUND.get(civs[1], "#FFFFFF")
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
