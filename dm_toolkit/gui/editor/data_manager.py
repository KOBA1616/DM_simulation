# -*- coding: utf-8 -*-
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtCore import QModelIndex
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel
from dm_toolkit.gui.editor.models.serializer import ModelSerializer
from dm_toolkit.gui.editor.services.feature_service import EditorFeatureService
from pydantic import BaseModel

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Orchestrates ModelSerializer and EditorFeatureService.
    """

    def __init__(self, model: QStandardItemModel):
        self.model = model
        self.serializer = ModelSerializer()
        self.feature_service = EditorFeatureService(model, self.serializer)

    def get_item_type(self, index_or_item):
        item = self.serializer._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_TYPE)
        return None

    def get_item_data(self, index_or_item):
        """
        Returns data as a dictionary (for compatibility).
        """
        return self.serializer.get_item_data(index_or_item)

    def get_item_model(self, index_or_item):
        """
        Returns the raw data object stored in the item (preferably a Pydantic model).
        """
        return self.serializer.get_item_model_obj(index_or_item)

    def set_item_data(self, index_or_item, data):
        """
        Sets data to the item.
        """
        self.serializer.set_item_data(index_or_item, data)

    def get_item_path(self, index_or_item):
        """Get hierarchical path of an item for identification."""
        item = self.serializer._ensure_item(index_or_item)
        if not item:
            return ""
        
        path_parts = []
        current = item
        while current:
            # Use item data to build unique path (e.g., card_id, effect_id, etc.)
            data = self.get_item_data(current) # Use getter to handle model->dict
            item_type = current.data(ROLE_TYPE)
            
            # Build identifier based on type and data
            if item_type == "card":
                identifier = f"card_{data.get('id', 'unknown')}"
            elif item_type == "effect":
                identifier = f"effect_{data.get('id', current.row())}"
            elif item_type == "command":
                identifier = f"command_{data.get('type', 'unknown')}_{current.row()}"
            else:
                identifier = f"{item_type or 'item'}_{current.row()}"
            
            path_parts.insert(0, identifier)
            current = current.parent()
        
        return "/".join(path_parts)

    def load_data(self, cards_data):
        self.serializer.load_data(self.model, cards_data)

    def get_full_data(self):
        return self.serializer.get_full_data(self.model)

    def reconstruct_card_model(self, card_item) -> CardModel:
        return self.serializer.reconstruct_card_model(card_item)

    # --- Feature Service Wrappers ---

    def add_new_card(self):
        return self.feature_service.add_new_card()

    def add_spell_side_item(self, card_item):
        return self.feature_service.add_spell_side_item(card_item)

    def remove_spell_side_item(self, card_item):
        self.feature_service.remove_spell_side_item(card_item)

    def add_reaction(self, parent_index):
        return self.feature_service.add_reaction(parent_index)

    def apply_template_by_key(self, card_item, template_key, display_label=None):
        return self.feature_service.apply_template_by_key(card_item, template_key, display_label)

    def remove_logic_by_label(self, card_item, label_substring):
        self.feature_service.remove_logic_by_label(card_item, label_substring)

    def add_option_slots(self, parent_item, count):
        self.feature_service.add_option_slots(parent_item, count)

    # --- Utility Wrappers or Remaining Logic ---

    def get_card_context_type(self, item):
        """Get the 'type' field of a card or spell_side item."""
        item = self.serializer._ensure_item(item)
        if not item:
            return "CREATURE"
        data = self.get_item_data(item)
        return data.get('type', 'CREATURE')

    def create_default_trigger_data(self):
        """Create default data for a triggered effect."""
        model = EffectModel(
            trigger="ON_PLAY",
            condition=None,
            commands=[]
        )
        return model.model_dump()

    def create_default_static_data(self):
        """Create default data for a static ability."""
        model = ModifierModel(
            type="COST_MODIFIER",
            value=0,
            scope="ALL"
        )
        return model.model_dump()

    def create_default_command_data(self, cmd_data=None):
        """Create or copy default command data."""
        if cmd_data:
            return copy.deepcopy(cmd_data)
        model = CommandModel(
            type="DRAW",
            amount=1
        )
        return model.model_dump()

    def format_command_label(self, cmd_data):
        """Format a label for a command item."""
        cmd_type = cmd_data.get('type', 'UNKNOWN')
        return f"{tr('Action')}: {tr(cmd_type)}"

    def add_child_item(self, parent_index, item_type, data, label):
        if isinstance(parent_index, QModelIndex):
            parent_item = self.model.itemFromIndex(parent_index)
        else:
            parent_item = parent_index
        return self.serializer.add_child_item(parent_item, item_type, data, label)

    def create_command_item(self, model):
        return self.serializer.create_command_item(model)

    def convert_action_tree_to_command(self, action_item):
        return self.serializer.convert_action_tree_to_command(action_item)
