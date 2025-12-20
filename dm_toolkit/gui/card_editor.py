import json
import os
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QWidget, QMessageBox, QToolBar, QFileDialog,
    QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from dm_toolkit.gui.editor.logic_tree import LogicTreeWidget
from dm_toolkit.gui.editor.property_inspector import PropertyInspector
from dm_toolkit.gui.editor.preview_pane import CardPreviewWidget
from dm_toolkit.gui.localization import tr

class CardEditor(QMainWindow):
    def __init__(self, json_path):
        super().__init__()
        self.json_path = json_path
        self.setWindowTitle(tr("Card Editor Ver 2.0"))
        self.resize(1400, 800) # Increased width for 3 panes

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

        # Update Preview Button (Right side)
        empty = QWidget()
        empty.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(empty)

        update_preview_act = QAction(tr("Update Preview"), self)
        update_preview_act.triggered.connect(self.update_preview_manual)
        toolbar.addAction(update_preview_act)

        # Splitter Layout (3 Panes)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Pane 1: Logic Tree
        self.tree_widget = LogicTreeWidget()

        # Pane 2: Property Inspector
        self.inspector = PropertyInspector()

        # Pane 3: Card Preview
        self.preview_widget = CardPreviewWidget()

        self.splitter.addWidget(self.tree_widget)
        self.splitter.addWidget(self.inspector)
        self.splitter.addWidget(self.preview_widget)

        # Adjust stretch factors
        self.splitter.setStretchFactor(0, 1) # Tree
        self.splitter.setStretchFactor(1, 2) # Inspector
        self.splitter.setStretchFactor(2, 1) # Preview

        self.setCentralWidget(self.splitter)

        # Signals
        self.tree_widget.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # Connect Data Changes from Inspector to Preview
        self.inspector.card_form.dataChanged.connect(self.on_data_changed)
        self.inspector.effect_form.dataChanged.connect(self.on_data_changed)
        self.inspector.action_form.dataChanged.connect(self.on_data_changed)
        self.inspector.spell_side_form.dataChanged.connect(self.on_data_changed)

        # Connect Structural Changes
        self.inspector.structure_update_requested.connect(self.on_structure_update)

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

            # Update Preview
            self.update_current_preview()

            # Auto-expand if it's a card and not already expanded
            item = self.tree_widget.model.itemFromIndex(index)
            if item:
                type_ = item.data(Qt.ItemDataRole.UserRole + 1)
                if type_ == "CARD":
                    self.tree_widget.expand(index)
        else:
            self.inspector.set_selection(None)
            self.preview_widget.clear_preview()

    def on_data_changed(self):
        # Refresh preview based on current selection
        self.update_current_preview()

    def update_preview_manual(self):
        self.on_data_changed()

    def update_current_preview(self):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid():
            self.preview_widget.clear_preview()
            return

        item = self.tree_widget.model.itemFromIndex(idx)

        # Find card item
        card_item = item
        while card_item:
            type_ = card_item.data(Qt.ItemDataRole.UserRole + 1)
            if type_ == "CARD":
                break
            card_item = card_item.parent()

        if card_item:
            # Reconstruct data to ensure structure (effects/actions) is up to date
            fresh_data = self.tree_widget.data_manager.reconstruct_card_data(card_item)
            if fresh_data:
                self.preview_widget.render_card(fresh_data)
            else:
                self.preview_widget.clear_preview()
        else:
            self.preview_widget.clear_preview()

    def on_structure_update(self, command, payload):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid(): return

        # Ensure we are operating on the Card Item
        item = self.tree_widget.model.itemFromIndex(idx)
        card_item = None

        item_type = item.data(Qt.ItemDataRole.UserRole + 1)
        if item_type == "CARD":
            card_item = item
        elif item_type in ["EFFECT", "SPELL_SIDE"]:
            card_item = item.parent()
        elif item_type == "ACTION":
            card_item = item.parent().parent()

        if not card_item: return

        if command == "ADD_SPELL_SIDE":
            self.tree_widget.add_spell_side(card_item.index())
            self.tree_widget.expand(card_item.index())
        elif command == "REMOVE_SPELL_SIDE":
            self.tree_widget.remove_spell_side(card_item.index())
        elif command == "ADD_REV_CHANGE":
            self.tree_widget.add_rev_change(card_item.index())
            self.tree_widget.expand(card_item.index())
        elif command == "REMOVE_REV_CHANGE":
            self.tree_widget.remove_rev_change(card_item.index())
        elif command == "GENERATE_OPTIONS":
            count = payload.get('count', 1)
            # Find the actual Action Item from the current selection
            action_item = None
            if item_type == "ACTION":
                 action_item = item

            if action_item:
                 self.tree_widget.data_manager.add_option_slots(action_item, count)
                 self.tree_widget.expand(action_item.index())
        elif command == "MOVE_EFFECT":
             item_obj = payload.get('item')
             target_type = payload.get('target_type')
             if item_obj and target_type:
                 self.tree_widget.move_effect_item(item_obj, target_type)

    def new_card(self):
        self.tree_widget.add_new_card()

    def add_effect(self):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid(): return
        
        item = self.tree_widget.model.itemFromIndex(idx)
        type_ = item.data(Qt.ItemDataRole.UserRole + 1)
        
        target_item = None
        if type_ == "CARD":
            target_item = item
        elif type_ == "SPELL_SIDE":
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
        elif type_ == "OPTION":
            target_item = item

        if target_item:
            new_act = {"type": "DESTROY", "scope": "TARGET_SELECT", "value1": 0, "filter": {}}
            label = f"{tr('Action')}: {tr('DESTROY')}"
            self.tree_widget.add_child_item(target_item.index(), "ACTION", new_act, label)

    def delete_item(self):
        self.tree_widget.remove_current_item()
