import json
import os
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QWidget, QMessageBox, QToolBar, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from dm_toolkit.gui.editor.logic_tree import LogicTreeWidget
from dm_toolkit.gui.editor.property_inspector import PropertyInspector
from dm_toolkit.gui.localization import tr

class CardEditor(QMainWindow):
    def __init__(self, json_path):
        super().__init__()
        self.json_path = json_path
        self.setWindowTitle(tr("Card Editor Ver 2.0"))
        self.resize(1200, 800)

        self.cards_data = []
        self.init_ui()
        self.load_data()

    def init_ui(self):
        # Toolbar
        toolbar = QToolBar(tr("Main Toolbar"))
        self.addToolBar(toolbar)

        new_act = QAction(tr("New Card"), self)
        new_act.triggered.connect(self.new_card)
        toolbar.addAction(new_act)

        save_act = QAction(tr("Save JSON"), self)
        save_act.triggered.connect(self.save_data)
        toolbar.addAction(save_act)

        add_eff_act = QAction(tr("Add Effect"), self)
        add_eff_act.triggered.connect(self.add_effect)
        toolbar.addAction(add_eff_act)

        add_act_act = QAction(tr("Add Action"), self)
        add_act_act.triggered.connect(self.add_action)
        toolbar.addAction(add_act_act)

        del_act = QAction(tr("Delete Item"), self)
        del_act.triggered.connect(self.delete_item)
        toolbar.addAction(del_act)

        # Splitter Layout
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.tree_widget = LogicTreeWidget()
        self.inspector = PropertyInspector()

        self.splitter.addWidget(self.tree_widget)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)

        self.setCentralWidget(self.splitter)

        # Signals
        self.tree_widget.selectionModel().selectionChanged.connect(self.on_selection_changed)

    def load_data(self):
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    self.cards_data = json.load(f)
            except Exception as e:
                QMessageBox.critical(self, tr("Error"), f"{tr('Failed to load JSON')}: {e}")
                self.cards_data = []
        else:
            self.cards_data = []

        self.tree_widget.load_data(self.cards_data)

    def save_data(self):
        data = self.tree_widget.get_full_data_from_model()
        try:
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, tr("Success"), tr("Cards saved successfully!"))
        except Exception as e:
            QMessageBox.critical(self, tr("Error"), f"{tr('Failed to save JSON')}: {e}")

    def on_selection_changed(self, selected, deselected):
        indexes = selected.indexes()
        if indexes:
            index = indexes[0]
            self.inspector.set_selection(index)
            # Auto-expand if it's a card and not already expanded
            item = self.tree_widget.model.itemFromIndex(index)
            if item:
                type_ = item.data(Qt.ItemDataRole.UserRole + 1)
                if type_ == "CARD":
                    self.tree_widget.expand(index)
        else:
            self.inspector.set_selection(None)

    def new_card(self):
        self.tree_widget.add_new_card()

    def add_effect(self):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid(): return
        
        item = self.tree_widget.model.itemFromIndex(idx)
        type_ = item.data(Qt.ItemDataRole.UserRole + 1)
        
        # If Card selected, add Effect
        # If Effect selected, add Sibling Effect (or nothing?) -> Let's support adding child to Card
        target_item = None
        if type_ == "CARD":
            target_item = item
        elif type_ == "EFFECT":
            target_item = item.parent()
        elif type_ == "ACTION":
            target_item = item.parent().parent()
            
        if target_item:
            new_eff = {"trigger": "ON_PLAY", "condition": {"type": "NONE"}, "actions": []}
            label = f"{tr('Effect')}: {tr('ON_PLAY')}"
            self.tree_widget.add_child_item(target_item.index(), "EFFECT", new_eff, label)

    def add_action(self):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid(): return
        
        item = self.tree_widget.model.itemFromIndex(idx)
        type_ = item.data(Qt.ItemDataRole.UserRole + 1)
        
        target_item = None
        if type_ == "EFFECT":
            target_item = item
        elif type_ == "ACTION":
            target_item = item.parent()

        if target_item:
            new_act = {"type": "DESTROY", "scope": "TARGET_SELECT", "value1": 0, "filter": {}}
            label = f"{tr('Action')}: {tr('DESTROY')}"
            self.tree_widget.add_child_item(target_item.index(), "ACTION", new_act, label)

    def delete_item(self):
        self.tree_widget.remove_current_item()
