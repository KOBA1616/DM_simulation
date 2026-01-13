# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.card_helpers import get_card_civilizations
from dm_toolkit.gui.editor.preview_pane import CardPreviewWidget

class CardDetailPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        # Replace simple labels with preview-style card widget
        self.preview = CardPreviewWidget()
        layout.addWidget(self.preview)

    def update_card(self, card_data, civ_map=None):
        """Render the hovered/selected card using the preview card widget."""
        if not card_data:
            self.preview.clear_preview()
            return

        civs = get_card_civilizations(card_data)
        # Build a minimal dict compatible with CardPreviewWidget
        t = str(getattr(card_data, 'type', 'CREATURE')).split('.')[-1]
        race = None
        if hasattr(card_data, 'races') and card_data.races:
            race = card_data.races[0]
        data = {
            'id': getattr(card_data, 'id', -1),
            'name': getattr(card_data, 'name', 'Unknown'),
            'cost': getattr(card_data, 'cost', 0),
            'power': getattr(card_data, 'power', 0),
            'civilizations': civs,
            'race': race,
            'type': t,
            'effects': [],
            'triggers': [],
            'spell_side': None,
        }
        try:
            self.preview.render_card(data)
        except Exception:
            # Fallback: clear if rendering fails
            self.preview.clear_preview()
