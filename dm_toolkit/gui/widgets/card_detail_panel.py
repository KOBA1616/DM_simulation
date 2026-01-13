# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.card_helpers import convert_card_data_to_dict
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

        # Use helper to convert card_data (C++ object) to dict compatible with TextGenerator
        try:
            data = convert_card_data_to_dict(card_data)
            self.preview.render_card(data)
        except Exception as e:
            # print(f"Error rendering card preview: {e}")
            # Fallback: clear if rendering fails
            self.preview.clear_preview()
