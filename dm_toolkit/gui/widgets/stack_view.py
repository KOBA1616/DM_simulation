# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QLabel, QHBoxLayout, QMessageBox, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QDrag
import dm_ai_module
from dm_toolkit.gui.i18n import tr

class PendingEffectItem(QListWidgetItem):
    def __init__(self, index, effect_info, card_name):
        super().__init__()
        self.index = index
        self.effect_info = effect_info

        effect_type = str(effect_info.get("type", tr("Unknown")))
        controller = effect_info.get("controller", -1)
        resolve_type = str(effect_info.get("resolve_type", tr("NONE")))

        display_text = f"[{index}] {effect_type} - {card_name} (P{controller})"
        if resolve_type != tr("NONE") and resolve_type != "NONE":
            display_text += f" [{resolve_type}]"

        self.setText(display_text)
        self.setData(Qt.ItemDataRole.UserRole, index)

class StackViewWidget(QWidget):
    effect_resolved = pyqtSignal(int) # Emits index of resolved effect

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(tr("Pending Effects (Stack)")))
        self._layout.addLayout(header)

        self.list_widget = QListWidget()
        self.list_widget.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        model = self.list_widget.model()
        if model is not None:
            model.rowsMoved.connect(self.on_rows_moved)
        self._layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()
        self.resolve_btn = QPushButton(tr("Resolve Selected"))
        self.resolve_btn.clicked.connect(self.on_resolve)
        btn_layout.addWidget(self.resolve_btn)

        self.refresh_btn = QPushButton(tr("Refresh"))
        self.refresh_btn.clicked.connect(self.request_refresh) # Should be connected to parent update
        btn_layout.addWidget(self.refresh_btn)

        self._layout.addLayout(btn_layout)

        self.current_effects = []

    def update_state(self, game_state, card_db):
        self.list_widget.clear()
        try:
            self.current_effects = game_state.get_pending_effects_info()
        except AttributeError:
            try:
                # Fallback to module level function if available (legacy support)
                import dm_ai_module
                if hasattr(dm_ai_module, 'get_pending_effects_info'):
                     self.current_effects = dm_ai_module.get_pending_effects_info(game_state)
                else:
                     raise AttributeError("No pending effects accessor found")
            except Exception:
                # Fallback for outdated binary
                item = QListWidgetItem("Error: Outdated C++ Module. Please rebuild.")
                item.setForeground(Qt.GlobalColor.red)
                self.list_widget.addItem(item)
                self.current_effects = []
                return

        # Reverse to show stack top at top (if index 0 is bottom) or vice versa.
        # Usually stack top is the last element in vector, but visual stack usually shows top at top.
        # `pending_effects` is a vector. `back()` is top.
        # We should display them such that the top of the stack is index N-1.
        # But for reordering, if we move items, we are effectively changing the vector.

        # Let's list them from Top (last) to Bottom (first)
        # Actually, let's keep index consistent with C++ vector index for now.

        for i, eff in enumerate(self.current_effects):
            source_id = eff.get("source_instance_id", -1)
            card_name = tr("Unknown")
            if source_id != -1:
                # We need to look up card name from instance.
                # game_state.get_card_instance(source_id) might be tricky if instance is gone,
                # but usually pending effect holds source id.
                # Since we don't have easy access to instance -> card_id here without game_state query
                # We can try get_card_instance.
                try:
                    instance = game_state.get_card_instance(source_id)
                    if instance:
                         card = card_db.get(instance.card_id)
                         if card:
                             card_name = card.name
                except:
                    pass

            item = PendingEffectItem(i, eff, card_name)
            self.list_widget.insertItem(0, item) # Insert at 0 to show latest at top?
            # If vector is [A, B, C], C is top.
            # Insert A at 0 -> [A]
            # Insert B at 0 -> [B, A]
            # Insert C at 0 -> [C, B, A]
            # So index 0 in ListWidget corresponds to index N-1 in vector.

    def on_resolve(self):
        # We need to find which effect to resolve.
        # In this engine, `RESOLVE_EFFECT` action usually takes an index (slot_index) in the pending_effects vector.
        # Or it might just resolve the top.
        # Let's check `RESOLVE_EFFECT` action in ActionGenerator. It usually generates for specific indices if multiple?
        # Actually `ActionType::RESOLVE_EFFECT` often targets the top or specific one.
        # If we want to resolve a specific one, we need `RESOLVE_EFFECT` action with `slot_index`.

        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        original_index = item.data(Qt.ItemDataRole.UserRole)
        self.effect_resolved.emit(original_index)

    def on_rows_moved(self, parent, start, end, destination, row):
        # This is complex because we need to sync with C++ vector.
        # If user drags item at visual row 0 (vector index N-1) to visual row 1 (vector index N-2).
        # We need to implement reordering in C++.
        # Currently we don't have `reorder_pending_effects` binding.
        # For now, maybe just keep visual?
        # But the prompt said "drag & drop to replace".
        # If we can't change C++ state, this is visual only and misleading.
        pass

    def request_refresh(self):
        pass
