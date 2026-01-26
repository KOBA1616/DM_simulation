# -*- coding: utf-8 -*-
import copy
from typing import Any, Optional, Dict
from pydantic import BaseModel

from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel
from dm_toolkit.gui.editor.models.serializer import ModelSerializer
from dm_toolkit.gui.editor.services.feature_service import EditorFeatureService
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Orchestrates ModelSerializer and EditorFeatureService.
    """

    def __init__(self, model: IEditorModel):
        self.model = model
        self.serializer = ModelSerializer()
        self.feature_service = EditorFeatureService(model, self.serializer)

    def get_item_type(self, index_or_item: Any) -> Optional[str]:
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_TYPE)
        return None

    def get_item_data(self, index_or_item: Any) -> Dict[str, Any]:
        """
        Returns data as a dictionary (for compatibility).
        """
        item = self._ensure_item(index_or_item)
        if item:
            return self.serializer.get_item_data(item)
        return {}

    def get_item_model(self, index_or_item: Any) -> Any:
        """
        Returns the raw data object stored in the item (preferably a Pydantic model).
        """
        item = self._ensure_item(index_or_item)
        if item:
            return self.serializer.get_item_model_obj(item)
        return None

    def set_item_data(self, index_or_item: Any, data: Any) -> None:
        """
        Sets data to the item.
        """
        item = self._ensure_item(index_or_item)
        if item:
            self.serializer.set_item_data(item, data)

    def get_item_path(self, index_or_item: Any) -> str:
        """Get hierarchical path of an item for identification."""
        item = self._ensure_item(index_or_item)
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

    def load_data(self, cards_data: list):
        self.serializer.load_data(self.model, cards_data)

    def get_full_data(self):
        return self.serializer.get_full_data(self.model)

    def reconstruct_card_model(self, card_item) -> CardModel:
        item = self._ensure_item(card_item)
        return self.serializer.reconstruct_card_model(item)

    # --- Feature Service Wrappers ---

    def add_new_card(self):
        return self.feature_service.add_new_card()

    def add_spell_side_item(self, card_item):
        item = self._ensure_item(card_item)
        return self.feature_service.add_spell_side_item(item)

    def remove_spell_side_item(self, card_item):
        item = self._ensure_item(card_item)
        self.feature_service.remove_spell_side_item(item)

    def add_reaction(self, parent_index):
        # parent_index can be index or item
        return self.feature_service.add_reaction(parent_index)

    def apply_template_by_key(self, card_item, template_key, display_label=None):
        item = self._ensure_item(card_item)
        return self.feature_service.apply_template_by_key(item, template_key, display_label)

    def remove_logic_by_label(self, card_item, label_substring):
        item = self._ensure_item(card_item)
        self.feature_service.remove_logic_by_label(item, label_substring)

    def add_option_slots(self, parent_item, count):
        item = self._ensure_item(parent_item)
        self.feature_service.add_option_slots(item, count)

    # --- Utility Wrappers or Remaining Logic ---

    def get_card_context_type(self, item):
        """Get the 'type' field of a card or spell_side item."""
        item = self._ensure_item(item)
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
        parent_item = self._ensure_item(parent_index)
        return self.serializer.add_child_item(parent_item, item_type, data, label, self.model)

    def create_command_item(self, model):
        return self.serializer.create_command_item(model, self.model)

    def add_command_contextual(self, index_or_item, cmd_data=None):
        """
        Adds a command item based on the context of the selected item.
        """
        item = self._ensure_item(index_or_item)
        if not item:
            return None

        type_ = self.get_item_type(item)
        target_item = None

        if type_ in ["EFFECT", "OPTION", "CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
            target_item = item
        elif type_ in ["COMMAND", "ACTION"]:
             cmd_model = self.get_item_model(item)
             cmd_type = None
             if hasattr(cmd_model, 'type'):
                 cmd_type = cmd_model.type
             elif isinstance(cmd_model, dict):
                 cmd_type = cmd_model.get('type')

             if cmd_type in ["IF", "IF_ELSE", "ELSE"]:
                 # Look for existing True branch
                 true_branch = None
                 false_branch = None
                 for i in range(item.rowCount()):
                     child = item.child(i)
                     role = child.data(ROLE_TYPE)
                     if role == "CMD_BRANCH_TRUE":
                         true_branch = child
                     elif role == "CMD_BRANCH_FALSE":
                         false_branch = child

                 if not true_branch:
                    true_branch = self.serializer.add_child_item(item, "CMD_BRANCH_TRUE", None, tr("If True"), self.model)

                 if cmd_type == "IF_ELSE" and not false_branch:
                    self.serializer.add_child_item(item, "CMD_BRANCH_FALSE", None, tr("If False"), self.model)

                 target_item = true_branch
             else:
                target_item = item.parent()
                if not target_item:
                    # Fallback for top-level items
                    target_item = self.model.invisibleRootItem()

        if not target_item:
            return None

        target_type = self.get_item_type(target_item)
        forbidden_types = ["CARD", "SPELL_SIDE", "MODIFIER", "REACTION_ABILITY", "KEYWORDS"]

        if target_type in forbidden_types:
             return None

        data = self.create_default_command_data(cmd_data)
        label = self.format_command_label(data)

        return self.add_child_item(target_item, "COMMAND", data, label)

    def update_effect_type(self, index_or_item: Any, target_type: str) -> None:
        item = self._ensure_item(index_or_item)
        if not item: return

        # Determine new role and label prefix
        new_role = "EFFECT"
        prefix = tr("Effect")

        # Simple mapping based on known types in LogicTreeWidget
        if target_type == "TRIGGERED":
             new_role = "EFFECT"
             prefix = tr("Effect")
        elif target_type == "STATIC":
             new_role = "MODIFIER"
             prefix = tr("Static")
        elif target_type == "REACTION":
             new_role = "REACTION_ABILITY"
             prefix = tr("Reaction")

        # Update Role
        item.setData(new_role, ROLE_TYPE)

        # Update Label (Keep content after colon if possible)
        current_text = item.text()
        if ":" in current_text:
            suffix = current_text.split(":", 1)[1]
            item.setText(f"{prefix}:{suffix}")
        else:
            item.setText(f"{prefix}: {target_type}")

    def _update_card_from_child(self, index_or_item: Any) -> None:
        # Placeholder
        pass

    # --- Helper ---
    def _ensure_item(self, index_or_item: Any) -> Optional[IEditorItem]:
        if isinstance(index_or_item, IEditorItem):
            return index_or_item
        # Use model to convert index
        return self.model.itemFromIndex(index_or_item)

    # --- Legacy Action conversion helpers (for tests / migration) ---

    def convert_action_tree_to_command(self, action_item):
        from dm_toolkit.gui.editor.action_converter import ActionConverter
        # Need to handle action_item as IEditorItem
        item = self._ensure_item(action_item)
        if not item: return {"type": "NONE"}

        try:
            action_data = item.data(ROLE_DATA)
        except Exception:
            action_data = None

        if not isinstance(action_data, dict):
            return {"type": "NONE", "legacy_warning": True, "str_param": "Invalid action item"}

        return ActionConverter.convert(action_data)

    def collect_conversion_preview(self, root_item):
        """Collect a preview list of action->command conversions under root_item."""
        previews = []
        root = self._ensure_item(root_item)

        def walk(item):
            if item is None:
                return

            try:
                role_type = item.data(ROLE_TYPE)
            except Exception:
                role_type = None

            if role_type == "ACTION":
                try:
                    label = item.text()
                except Exception:
                    label = ""
                cmd_data = self.convert_action_tree_to_command(item)
                previews.append({"label": label, "cmd_data": cmd_data})

            try:
                rc = item.rowCount()
            except Exception:
                rc = 0
            for i in range(rc):
                try:
                    child = item.child(i)
                except Exception:
                    child = None
                walk(child)

        walk(root)
        return previews
