# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QScrollArea, QWidget
from PyQt6.QtCore import Qt
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.widgets.zone_widget import ZoneWidget

class ZonePopup(QDialog):
    def __init__(self, title, card_data_list, card_db, civ_map=None, legal_actions=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 400)
        self.card_data_list = card_data_list
        self.card_db = card_db
        self.civ_map = civ_map
        self.legal_actions = legal_actions if legal_actions else []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Scroll Area for ZoneWidget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.zone_widget = ZoneWidget(self.windowTitle(), parent=self)
        scroll_area.setWidget(self.zone_widget)

        layout.addWidget(scroll_area)

        close_btn = QPushButton(tr("Close"))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_content(self.card_data_list, self.card_db, self.civ_map, self.legal_actions)

    def update_content(self, card_data_list, card_db, civ_map=None, legal_actions=None):
        self.card_data_list = card_data_list
        self.card_db = card_db
        self.civ_map = civ_map
        if legal_actions:
            self.legal_actions = legal_actions

        self.zone_widget.update_cards(
            self.card_data_list,
            self.card_db,
            civ_map=self.civ_map,
            legal_actions=self.legal_actions,
            collapsed=False
        )
