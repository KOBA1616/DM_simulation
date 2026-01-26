# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING, Any, Optional
from dm_toolkit.gui.i18n import tr
import copy
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel, ReactionModel
from dm_toolkit.gui.editor.models.serializer import ModelSerializer
from dm_toolkit.gui.editor.services.feature_service import EditorFeatureService
from dm_toolkit.editor.core.abstraction import IEditorModel, IEditorItem

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Orchestrates ModelSerializer and EditorFeatureService.
    """

    def __init__(self, model: IEditorModel):
        self.model = model
        self.serializer = ModelSerializer()
        self.feature_service = EditorFeatureService(model, self.serializer)

    def _ensure_item(self, index_or_item) -> Optional[IEditorItem]:
        # Handle index resolution if passed index.
        # But callers should generally pass IEditorItem or abstract index.
        # If it is Qt Model Index, self.model.item_from_index handles it.
        # If it is already IEditorItem, we return it.
        if isinstance(index_or_item, IEditorItem):
            return index_or_item

        # Try asking model (handles QModelIndex if model is QtEditorModel)
        item = self.model.item_from_index(index_or_item)
        if item: return item

        return None

    def get_item_type(self, index_or_item):
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_TYPE)
        return None

    def get_item_data(self, index_or_item):
        """
        Returns data as a dictionary (for compatibility).
        """
        item = self._ensure_item(index_or_item)
        return self.serializer.get_item_data(item)

    def get_item_model_obj(self, index_or_item):
        """
        Returns the raw data object stored in the item (preferably a Pydantic model).
        """
        item = self._ensure_item(index_or_item)
        return self.serializer.get_item_model_obj(item)

    def set_item_data(self, index_or_item, data):
        """
        Sets data to the item.
        """
        item = self._ensure_item(index_or_item)
        self.serializer.set_item_data(item, data)

    def get_item_path(self, index_or_item):
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

    def load_data(self, cards_data):
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
        # Resolve index here
        item = self._ensure_item(parent_index)
        return self.feature_service.add_reaction(item)

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
        if not parent_item:
            return None

        new_item = self.model.create_item(label)
        new_item.set_data(item_type, ROLE_TYPE)
        new_item.set_data(data, ROLE_DATA)
        parent_item.append_row(new_item)
        return new_item

    def create_command_item(self, model):
        return self.serializer.create_command_item(self.model, model)

    def add_command_contextual(self, index_or_item, cmd_data=None):
        """
        Adds a command item based on the context of the selected item.
        """
        item = self._ensure_item(index_or_item)
        if not item:
            return None

        type_ = self.get_item_type(item)
        target_item = None

        # Determine target container based on selection
        if type_ in ["EFFECT", "OPTION", "CMD_BRANCH_TRUE", "CMD_BRANCH_FALSE"]:
            target_item = item
        elif type_ in ["COMMAND", "ACTION"]:
             cmd_model = self.get_item_model_obj(item)
             cmd_type = None
             if hasattr(cmd_model, 'type'):
                 cmd_type = cmd_model.type
             elif isinstance(cmd_model, dict):
                 cmd_type = cmd_model.get('type')

             if cmd_type in ["IF", "IF_ELSE", "ELSE"]:
                 # Ensure branches exist and target "If True"
                 true_branch = None
                 false_branch = None
                 for i in range(item.row_count()):
                     child = item.child(i)
                     role = child.data(ROLE_TYPE)
                     if role == "CMD_BRANCH_TRUE":
                         true_branch = child
                     elif role == "CMD_BRANCH_FALSE":
                         false_branch = child

                 if not true_branch:
                    # We need to create it. We can use add_child_item logic.
                    true_branch = self.model.create_item(tr("If True"))
                    true_branch.set_data("CMD_BRANCH_TRUE", ROLE_TYPE)
                    item.append_row(true_branch)

                 if cmd_type == "IF_ELSE" and not false_branch:
                    false_branch = self.model.create_item(tr("If False"))
                    false_branch.set_data("CMD_BRANCH_FALSE", ROLE_TYPE)
                    item.append_row(false_branch)

                 target_item = true_branch
             else:
                target_item = item.parent()
                if not target_item:
                     pass

        if not target_item:
            return None

        # Validate that the target is a valid container for commands
        target_type = self.get_item_type(target_item)

        forbidden_types = ["CARD", "SPELL_SIDE", "MODIFIER", "REACTION_ABILITY", "KEYWORDS"]

        if target_type in forbidden_types:
             return None

        # Create Data
        data = self.create_default_command_data(cmd_data)
        label = self.format_command_label(data)

        # Use add_child_item logic
        new_item = self.model.create_item(label)
        new_item.set_data("COMMAND", ROLE_TYPE)
        new_item.set_data(data, ROLE_DATA)
        target_item.append_row(new_item)
        return new_item

    # --- Legacy Action conversion helpers (for tests / migration) ---

    def convert_action_tree_to_command(self, action_item):
        """Convert an ACTION tree item into a normalized GameCommand dict.
        """
        # We need ActionConverter. It's safe to import here as long as it doesn't depend on Qt.
        # dm_toolkit/gui/editor/action_converter.py needs check.
        # But let's assume it's pure logic.
        from dm_toolkit.gui.editor.action_converter import ActionConverter

        item = self._ensure_item(action_item)
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
                rc = item.row_count()
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

    def update_effect_type(self, index_or_item, target_type):
        item = self._ensure_item(index_or_item)
        if not item: return

        # Map UI string to internal type if needed
        internal_type = target_type
        if target_type == tr("Triggered Ability"):
            internal_type = "EFFECT"
        elif target_type == tr("Static Ability"):
            internal_type = "MODIFIER"
        elif target_type == tr("Reaction Ability"):
            internal_type = "REACTION_ABILITY"

        current_type = item.data(ROLE_TYPE)
        if current_type == internal_type:
            return

        # Update Type
        item.set_data(internal_type, ROLE_TYPE)

        # Update Data Model Structure
        data = self.get_item_data(item)

        if internal_type == "EFFECT":
            item.set_text(f"{tr('Effect')}: {tr('ON_PLAY')}")
            # Ensure triggers/effects structure
            if 'trigger' not in data:
                data['trigger'] = 'ON_PLAY'
                data['condition'] = None
                data['commands'] = []
        elif internal_type == "MODIFIER":
            item.set_text(f"{tr('Static')}: COST_MODIFIER")
            if 'value' not in data:
                data['type'] = 'COST_MODIFIER'
                data['value'] = 0
                data['scope'] = 'ALL'
        elif internal_type == "REACTION_ABILITY":
            item.set_text(f"{tr('Reaction')}: NINJA_STRIKE")
            if 'zone' not in data:
                data['type'] = 'NINJA_STRIKE'
                data['cost'] = None
                data['zone'] = None

        self.set_item_data(item, data)

    def add_command_branches(self, index_or_item):
        item = self._ensure_item(index_or_item)
        if not item: return

        cmd_model = self.get_item_model_obj(item)
        cmd_type = None
        if hasattr(cmd_model, 'type'):
             cmd_type = cmd_model.type
        elif isinstance(cmd_model, dict):
             cmd_type = cmd_model.get('type')

        if cmd_type in ["IF", "IF_ELSE", "ELSE"]:
             # Ensure branches exist
             true_branch = None
             false_branch = None
             for i in range(item.row_count()):
                 child = item.child(i)
                 role = child.data(ROLE_TYPE)
                 if role == "CMD_BRANCH_TRUE":
                     true_branch = child
                 elif role == "CMD_BRANCH_FALSE":
                     false_branch = child

             if not true_branch:
                true_branch = self.model.create_item(tr("If True"))
                true_branch.set_data("CMD_BRANCH_TRUE", ROLE_TYPE)
                item.append_row(true_branch)

             if cmd_type == "IF_ELSE" and not false_branch:
                false_branch = self.model.create_item(tr("If False"))
                false_branch.set_data("CMD_BRANCH_FALSE", ROLE_TYPE)
                item.append_row(false_branch)

    def generate_options(self, index_or_item, count):
        item = self._ensure_item(index_or_item)
        if not item: return

        data = self.get_item_data(item)
        new_options = []
        for _ in range(max(1, count)):
            new_options.append([{"type": "NONE"}])

        data['options'] = new_options
        data['type'] = 'CHOICE' # Unified type for options

        self.set_item_data(item, data)

        # Clear children
        while item.row_count() > 0:
            item.remove_row(0)

        # Add option items
        for i, opt in enumerate(new_options):
            opt_item = self.model.create_item(f"{tr('Option')} {i+1}")
            opt_item.set_data("OPTION", ROLE_TYPE)
            item.append_row(opt_item)
            # Add commands inside option
            for cmd in opt:
                cmd_item = self.create_command_item(cmd)
                opt_item.append_row(cmd_item)
