# -*- coding: utf-8 -*-
import json
import os
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QVBoxLayout, QWidget, QMessageBox, QToolBar, QFileDialog,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QAction, QKeySequence, QStandardItem
from dm_toolkit.gui.editor.logic_tree import LogicTreeWidget
from dm_toolkit.gui.editor.property_inspector import PropertyInspector
from dm_toolkit.gui.editor.preview_pane import CardPreviewWidget
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.consts import (
    STRUCT_CMD_ADD_CHILD_EFFECT, STRUCT_CMD_ADD_SPELL_SIDE, STRUCT_CMD_REMOVE_SPELL_SIDE,
    STRUCT_CMD_ADD_REV_CHANGE, STRUCT_CMD_REMOVE_REV_CHANGE, 
    STRUCT_CMD_ADD_MEKRAID, STRUCT_CMD_REMOVE_MEKRAID,
    STRUCT_CMD_ADD_FRIEND_BURST, STRUCT_CMD_REMOVE_FRIEND_BURST,
    STRUCT_CMD_ADD_MEGA_LAST_BURST, STRUCT_CMD_REMOVE_MEGA_LAST_BURST,
    STRUCT_CMD_GENERATE_BRANCHES, STRUCT_CMD_GENERATE_OPTIONS, STRUCT_CMD_MOVE_EFFECT, 
    STRUCT_CMD_ADD_CHILD_ACTION, STRUCT_CMD_REPLACE_WITH_COMMAND
)

class CardEditor(QMainWindow):
    data_saved = pyqtSignal()
    def __init__(self, json_path):
        super().__init__()
        self.json_path = json_path
        self.setWindowTitle(tr("Card Editor Ver 2.0"))
        self.resize(1600, 900) # Optimized for 3-pane layout with room for OS/DE chrome

        self.cards_data = []
        self.init_ui()
        self.load_data()

    def init_ui(self):
        # Toolbar
        toolbar = QToolBar(tr("Main Toolbar"))
        self.addToolBar(toolbar)
        toolbar.setIconSize(QSize(16, 16))  # Compact icon size
        toolbar.setStyleSheet("QToolBar { padding: 2px; }")

        new_act = QAction(tr("New Card"), self)
        new_act.triggered.connect(self.new_card)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.setStatusTip(tr("Create a new card"))
        toolbar.addAction(new_act)

        save_act = QAction(tr("Save JSON"), self)
        save_act.triggered.connect(self.save_data)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.setStatusTip(tr("Save all changes to JSON"))
        toolbar.addAction(save_act)

        add_eff_act = QAction(tr("Add Effect"), self)
        add_eff_act.triggered.connect(self.add_effect)
        add_eff_act.setShortcut("Ctrl+Shift+E")
        add_eff_act.setText(tr("Add Eff"))
        add_eff_act.setStatusTip(tr("Add a new effect to the selected card"))
        toolbar.addAction(add_eff_act)

        add_act_act = QAction(tr("Add Command"), self)
        add_act_act.triggered.connect(self.add_command)
        add_act_act.setShortcut("Ctrl+Shift+C")
        add_act_act.setText(tr("Add Cmd"))
        add_act_act.setStatusTip(tr("Add a command to the selected effect"))
        toolbar.addAction(add_act_act)

        del_act = QAction(tr("Delete Item"), self)
        del_act.triggered.connect(self.delete_item)
        del_act.setShortcut(QKeySequence.StandardKey.Delete)
        del_act.setText(tr("Delete"))
        del_act.setStatusTip(tr("Delete the selected item"))
        toolbar.addAction(del_act)

        # Update Preview Button (Right side)
        empty = QWidget()
        empty.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(empty)

        update_preview_act = QAction(tr("Update Preview"), self)
        update_preview_act.triggered.connect(self.update_preview_manual)
        update_preview_act.setShortcut(QKeySequence.StandardKey.Refresh)
        update_preview_act.setText(tr("Update"))
        update_preview_act.setStatusTip(tr("Force update the card preview"))
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
        # Optimized stretch factors for balanced 3-pane layout
        self.splitter.setStretchFactor(0, 1)   # Tree (25%)
        self.splitter.setStretchFactor(1, 2)   # Inspector (50%)
        self.splitter.setStretchFactor(2, 1)   # Preview (25%)
        
        # Set minimum widths to prevent panes from becoming too narrow
        self.tree_widget.setMinimumWidth(250)
        self.inspector.setMinimumWidth(400)
        self.preview_widget.setMinimumWidth(250)

        self.setCentralWidget(self.splitter)
        self.statusBar() # Ensure status bar is created

        # Signals
        sel = self.tree_widget.selectionModel()
        if sel is not None:
            sel.selectionChanged.connect(self.on_selection_changed)

        # Connect Data Changes from Inspector to Preview
        self.inspector.card_form.dataChanged.connect(self.on_data_changed)
        self.inspector.effect_form.dataChanged.connect(self.on_data_changed)
        # Unified form replaces previous action/command editors
        self.inspector.unified_form.dataChanged.connect(self.on_data_changed)
        self.inspector.spell_side_form.dataChanged.connect(self.on_data_changed)
        self.inspector.modifier_form.dataChanged.connect(self.on_data_changed)

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
            self.data_saved.emit()
            sb = self.statusBar()
            if sb is not None:
                sb.showMessage(tr("Cards saved successfully!"), 3000)
            # Also show a confirmation dialog so the user notices the save action
            try:
                QMessageBox.information(self, tr("Saved"), tr("Cards saved successfully!"))
            except Exception:
                # If running headless or dialogs fail, ignore
                pass
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
            item = self.tree_widget.standard_model.itemFromIndex(index)
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
        # Ensure internal canonical cache is updated for current selection
        try:
            idx = self.tree_widget.currentIndex()
            if idx.isValid():
                item = self.tree_widget.standard_model.itemFromIndex(idx)
                if item is not None:
                    # Propagate up to data_manager to refresh cached CIR
                    self.tree_widget.data_manager._update_card_from_child(item)
        except Exception:
            pass

    def update_preview_manual(self):
        self.on_data_changed()

    def update_current_preview(self):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid():
            self.preview_widget.clear_preview()
            return

        item = self.tree_widget.standard_model.itemFromIndex(idx)

        # Find card item
        card_item = item
        while card_item:
            type_ = card_item.data(Qt.ItemDataRole.UserRole + 1)
            if type_ == "CARD":
                break
            card_item = card_item.parent()

        if card_item:
            # Reconstruct data to ensure structure (effects/actions) is up to date
            fresh_model = self.tree_widget.data_manager.reconstruct_card_model(card_item)
            fresh_data = fresh_model.model_dump(by_alias=True) if hasattr(fresh_model, 'model_dump') else fresh_model.dict(by_alias=True)
            if fresh_data:
                self.preview_widget.render_card(fresh_data)
            else:
                self.preview_widget.clear_preview()
        else:
            self.preview_widget.clear_preview()

    def on_structure_update(self, command, payload):
        idx = self.tree_widget.currentIndex()

        # Determine context for updates that modify hierarchy
        if command == STRUCT_CMD_REPLACE_WITH_COMMAND:
            target_data = payload
            target_idx = idx

            # Handle new payload structure with explicit target item
            if 'target_item' in payload and 'new_data' in payload:
                target_item = payload['target_item']
                if target_item:
                    target_idx = target_item.index()
                target_data = payload['new_data']

            if target_idx.isValid():
                self.tree_widget.replace_item_with_command(target_idx, target_data)
            return

        if not idx.isValid(): return

        # Ensure we are operating on the Card Item
        item = self.tree_widget.standard_model.itemFromIndex(idx)
        if item is None:
            return
        card_item = None

        item_type = item.data(Qt.ItemDataRole.UserRole + 1)
        if item_type == "CARD":
            card_item = item
        elif item_type in ["EFFECT", "SPELL_SIDE"]:
            parent = item.parent()
            if parent is not None:
                card_item = parent
        elif item_type in ["ACTION", "COMMAND"]:
            parent = item.parent()
            if parent is not None:
                grand = parent.parent()
                if grand is not None:
                    card_item = grand

        if card_item is None:
            return

        if command == STRUCT_CMD_ADD_SPELL_SIDE:
            self.tree_widget.add_spell_side(card_item.index())
            self.tree_widget.expand(card_item.index())
        elif command == STRUCT_CMD_REMOVE_SPELL_SIDE:
            self.tree_widget.remove_spell_side(card_item.index())
        elif command == STRUCT_CMD_ADD_REV_CHANGE:
            self.tree_widget.add_rev_change(card_item.index())
            self.tree_widget.expand(card_item.index())
        elif command == STRUCT_CMD_REMOVE_REV_CHANGE:
            self.tree_widget.remove_rev_change(card_item.index())
        elif command == STRUCT_CMD_ADD_MEKRAID:
            self.tree_widget.add_mekraid(card_item.index())
            self.tree_widget.expand(card_item.index())
        elif command == STRUCT_CMD_REMOVE_MEKRAID:
            self.tree_widget.remove_mekraid(card_item.index())
        elif command == STRUCT_CMD_ADD_FRIEND_BURST:
            self.tree_widget.add_friend_burst(card_item.index())
            self.tree_widget.expand(card_item.index())
        elif command == STRUCT_CMD_REMOVE_FRIEND_BURST:
            self.tree_widget.remove_friend_burst(card_item.index())
        elif command == STRUCT_CMD_ADD_MEGA_LAST_BURST:
            self.tree_widget.add_mega_last_burst(card_item.index())
            self.tree_widget.expand(card_item.index())
        elif command == STRUCT_CMD_REMOVE_MEGA_LAST_BURST:
            self.tree_widget.remove_mega_last_burst(card_item.index())
        elif command == STRUCT_CMD_GENERATE_OPTIONS:
            count = payload.get('count', 1)
            # Find the actual Action Item from the current selection
            action_item = None
            if item_type in ["ACTION", "COMMAND"]:
                 action_item = item

            if action_item:
                 self.tree_widget.data_manager.add_option_slots(action_item, count)
                 self.tree_widget.expand(action_item.index())
        elif command == STRUCT_CMD_GENERATE_BRANCHES:
            self.tree_widget.generate_branches_for_current()
        elif command == STRUCT_CMD_MOVE_EFFECT:
             item_obj = payload.get('item')
             target_type = payload.get('target_type')
             if item_obj and target_type:
                 self.tree_widget.move_effect_item(item_obj, target_type)
        elif command == STRUCT_CMD_ADD_CHILD_EFFECT:
            eff_type = payload.get('type')
            if eff_type == "KEYWORDS":
                self.tree_widget.add_keywords(item.index())
            elif eff_type == "TRIGGERED":
                self.tree_widget.add_trigger(item.index())
            elif eff_type == "STATIC":
                self.tree_widget.add_static(item.index())
            elif eff_type == "REACTION":
                self.tree_widget.add_reaction(item.index())
        elif command == STRUCT_CMD_ADD_CHILD_ACTION:
            if item_type == "EFFECT":
                self.tree_widget.add_action_to_effect(item.index())
            elif item_type == "OPTION":
                self.tree_widget.add_action_to_option(item.index())
            elif item_type in ["ACTION", "COMMAND"]:
                self.tree_widget.add_action_sibling(item.index())

    def new_card(self):
        self.tree_widget.add_new_card()

    def add_effect(self):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid(): return
        
        item = self.tree_widget.standard_model.itemFromIndex(idx)
        if item is None:
            QMessageBox.warning(self, tr("Warning"), tr("Please select a Card or Effect group to add an Effect."))
            return

        # Traverse up to find CARD or SPELL_SIDE
        target_item: QStandardItem | None = item
        found = False
        while target_item is not None:
            type_ = target_item.data(Qt.ItemDataRole.UserRole + 1)
            if type_ in ["CARD", "SPELL_SIDE"]:
                found = True
                break
            parent = target_item.parent()
            if parent is None:
                target_item = None
            else:
                target_item = parent

        if found and target_item is not None:
            self.tree_widget.add_effect_interactive(target_item.index())
        else:
            QMessageBox.warning(self, tr("Warning"), tr("Please select a Card or Effect group to add an Effect."))

    def add_command(self):
        idx = self.tree_widget.currentIndex()
        if not idx.isValid(): return

        item = self.tree_widget.standard_model.itemFromIndex(idx)
        if item is None:
            QMessageBox.warning(self, tr("Warning"), tr("Please select an Effect or Option to add a Command."))
            return
        type_ = item.data(Qt.ItemDataRole.UserRole + 1)

        # Centralized logic in LogicTreeWidget
        valid_types = ["EFFECT", "OPTION", "COMMAND", "ACTION", "CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]
        if type_ in valid_types:
            self.tree_widget.add_command_contextual()
        else:
            QMessageBox.warning(self, tr("Warning"), tr("Please select an Effect or Option to add a Command."))

    def delete_item(self):
        self.tree_widget.remove_current_item()


def main(argv: list[str] | None = None) -> int:
    """Entry point for the card editor.

    This GUI is useful for UI review and card JSON editing and does not require
    the native dm_ai_module.
    """
    if argv is None:
        argv = sys.argv[1:]

    # Determine cards.json path
    json_path = argv[0] if len(argv) >= 1 else os.path.join('data', 'cards.json')
    if not os.path.exists(json_path):
        # Fallback to repo-root relative when launched from within package folder
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cand = os.path.join(base_dir, 'data', 'cards.json')
        if os.path.exists(cand):
            json_path = cand

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = CardEditor(json_path)
    win.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
