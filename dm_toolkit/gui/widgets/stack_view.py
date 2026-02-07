# -*- coding: utf-8 -*-
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QLabel, QHBoxLayout, QMessageBox, QAbstractItemView
)
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData
from PyQt6.QtGui import QDrag
import dm_ai_module
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.utils.card_helpers import get_card_name
import logging

logger = logging.getLogger(__name__)

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
            except Exception as e:
                # Fallback for outdated binary
                logger.warning(f"Failed to get pending effects: {e}")
                item = QListWidgetItem("Error: Outdated C++ Module. Please rebuild.")
                item.setForeground(Qt.GlobalColor.red)
                self.list_widget.addItem(item)
                self.current_effects = []
                return

        logger.debug(f"[StackView] Update state: {len(self.current_effects)} pending effects")
        
        # Display effects in reverse order: stack top (last element) appears at top of list
        # This provides intuitive visualization where the top-of-stack is visually on top
        # The UserRole data stores the actual vector index for C++ operations
        
        for i, eff in enumerate(self.current_effects):
            source_id = eff.get("source_instance_id", -1)
            card_name = tr("Unknown")
            if source_id != -1:
                try:
                    instance = game_state.get_card_instance(source_id)
                    if instance:
                         card = card_db.get(instance.card_id)
                         if card:
                             card_name = get_card_name(card)
                except:
                    pass

            # Store actual vector index for resolution
            item = PendingEffectItem(i, eff, card_name)
            logger.debug(f"[StackView] Adding effect {i}: {eff.get('type', 'UNKNOWN')} from {card_name}")
            self.list_widget.insertItem(0, item)  # Reverse insertion: vector[0] appears last (bottom)

    def visual_index_to_vector_index(self, visual_row: int) -> int:
        """Convert visual list index to pending_effects vector index."""
        if not self.current_effects:
            return -1
        vector_size = len(self.current_effects)
        # If we have N items and insert at 0, visual row 0 = vector index N-1
        return vector_size - 1 - visual_row

    def on_resolve(self):
        """Resolve the selected pending effect.
        
        Gets the vector index from the item's UserRole data, which was set during
        update_state() and remains the authoritative index for C++ operations.
        """
        logger.debug(f"[StackView] on_resolve called, button enabled={self.resolve_btn.isEnabled()}, list size={self.list_widget.count()}")
        selected_items = self.list_widget.selectedItems()
        logger.debug(f"[StackView] on_resolve called: {len(selected_items)} items selected")
        if not selected_items:
            logger.debug("[StackView] No items selected, checking all items in list:")
            for i in range(self.list_widget.count()):
                item = self.list_widget.item(i)
                logger.debug(f"  Item {i}: {item.text() if item else 'None'}")
            return

        item = selected_items[0]
        # UserRole contains the actual vector index from PendingEffectItem
        vector_index = item.data(Qt.ItemDataRole.UserRole)
        logger.debug(f"[StackView] Resolved effect at vector index: {vector_index}")
        if vector_index is not None:
            logger.debug(f"[StackView] Emitting effect_resolved signal with index {vector_index}")
            self.effect_resolved.emit(vector_index)

    def on_rows_moved(self, parent, start, end, destination, row):
        """Handle drag-and-drop reordering of effects.
        
        Currently only a placeholder: proper implementation would require
        reordering in the C++ pending_effects vector through a binding.
        """
        # TODO: Implement pending effect reordering through C++ binding if needed
        pass

    def request_refresh(self):
        pass
