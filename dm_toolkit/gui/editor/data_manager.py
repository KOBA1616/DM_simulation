# -*- coding: utf-8 -*-
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QModelIndex
from dm_toolkit.gui.localization import tr
import uuid
import json
import os
import copy
from typing import Optional, Type, TypeVar
from pydantic import BaseModel
from dm_toolkit.types import JSON
from dm_toolkit.gui.editor import normalize
from dm_toolkit.gui.editor.action_converter import ActionConverter, convert_action_to_objs
from dm_toolkit.gui.editor.command_model import CommandDef, WarningCommand
from dm_toolkit.gui.editor.templates import LogicTemplateManager
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import (
    CardModel, EffectModel, CommandModel, ModifierModel, ReactionModel,
    ConditionModel, FilterModel
)
from dm_toolkit.gui.editor.data_manager_helpers import set_item_model, get_item_model, ModelAwareItem

T = TypeVar('T')

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Handles loading, saving (reconstruction), and item creation (ID generation).
    Uses Pydantic data models for data integrity.
    """

    def __init__(self, model: QStandardItemModel):
        self.model = model
        self.template_manager = LogicTemplateManager.get_instance()

    def get_item_type(self, index_or_item):
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_TYPE)
        return None

    def get_item_data(self, index_or_item):
        """
        Returns the raw dict data stored in the item.
        For typed access, use get_item_model_data.
        """
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_DATA) or {}
        return {}

    def get_item_model_data(self, index_or_item, model_cls: Type[T]) -> Optional[T]:
        """
        Returns the data as a Pydantic model instance.
        """
        item = self._ensure_item(index_or_item)
        if item:
            return get_item_model(item, model_cls)
        return None

    def set_item_data(self, index_or_item, data):
        """
        Sets data to the item. 'data' can be a Pydantic model or a dict.
        Internally converts to dict for storage, but validates if model is passed.
        """
        item = self._ensure_item(index_or_item)
        if item:
            if hasattr(data, 'model_dump') or hasattr(data, 'dict'):
                set_item_model(item, data)
            else:
                # Fallback for direct dict usage (legacy or test)
                item.setData(data, ROLE_DATA)

    def get_item_path(self, index_or_item):
        """Get hierarchical path of an item for identification."""
        item = self._ensure_item(index_or_item)
        if not item:
            return ""
        
        path_parts = []
        current = item
        while current:
            # Use item data to build unique path (e.g., card_id, effect_id, etc.)
            data = current.data(ROLE_DATA) or {}
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

    def _ensure_item(self, index_or_item):
        if isinstance(index_or_item, QModelIndex):
            return self.model.itemFromIndex(index_or_item)
        return index_or_item

    def load_data(self, cards_data):
        self.model.clear()
        self.model.setHorizontalHeaderLabels([tr("Logic Tree")])

        for card_raw in cards_data:
            # --- Migration / Normalization Step ---
            # Handle legacy 'triggers' vs 'effects' normalization
            if 'triggers' in card_raw:
                card_raw['effects'] = card_raw.pop('triggers')

            # Strict migration: Lift actions to commands HERE
            if 'effects' in card_raw:
                for eff in card_raw['effects']:
                    self._lift_actions_to_commands(eff)

            # --- Validation Step ---
            try:
                card_model = CardModel(**card_raw)
            except Exception as e:
                # print(f"Model validation failed for card {card_raw.get('id')}: {e}")
                card_model = CardModel.construct(**card_raw)

            # --- Creation Step ---
            card_item = self._create_card_item(card_model)

            # Effects
            # Fallback for construct where effects might remain as list of dicts
            effects_list = card_model.effects or []
            if effects_list and isinstance(effects_list[0], dict):
                 new_list = []
                 for eff in effects_list:
                     if isinstance(eff, dict):
                         new_list.append(EffectModel(**eff))
                     else:
                         new_list.append(eff)
                 effects_list = new_list

            for effect in effects_list:
                eff_item = self._create_effect_item(effect)
                self._load_effect_children(eff_item, effect)
                card_item.appendRow(eff_item)

            # Static Abilities
            for modifier in card_model.static_abilities:
                mod_item = self._create_modifier_item(modifier)
                card_item.appendRow(mod_item)

            # Reaction Abilities
            for reaction in card_model.reaction_abilities:
                ra_item = self._create_reaction_item(reaction)
                card_item.appendRow(ra_item)

            # Spell Side
            if card_model.spell_side:
                spell_item = self._create_spell_side_item(card_model.spell_side)
                for effect in card_model.spell_side.effects:
                    eff_item = self._create_effect_item(effect)
                    self._load_effect_children(eff_item, effect)
                    spell_item.appendRow(eff_item)
                for modifier in card_model.spell_side.static_abilities:
                    mod_item = self._create_modifier_item(modifier)
                    spell_item.appendRow(mod_item)
                card_item.appendRow(spell_item)

            self.model.appendRow(card_item)
            self._sync_editor_warnings(card_item)

    def _load_effect_children(self, eff_item, effect_model: EffectModel):
        for command in effect_model.commands:
            cmd_item = self.create_command_item(command)
            eff_item.appendRow(cmd_item)

    def _lift_actions_to_commands(self, effect_data):
        """
        One-time migration helper used during load_data.
        Converts legacy 'actions' list to 'commands' in place.
        """
        if 'actions' in effect_data:
            legacy_actions = effect_data.pop('actions')
            commands = effect_data.get('commands', [])
            for act in legacy_actions:
                 try:
                     objs = convert_action_to_objs(act)
                     for o in objs:
                         if hasattr(o, 'to_dict'):
                             commands.append(o.to_dict())
                         elif isinstance(o, dict):
                             commands.append(o)
                         elif hasattr(o, 'dict'): # Pydantic v1
                             commands.append(o.dict())
                         elif hasattr(o, 'model_dump'): # Pydantic v2
                             commands.append(o.model_dump())
                 except:
                     pass
            effect_data['commands'] = commands

    def get_full_data(self):
        cards = []
        root = self.model.invisibleRootItem()
        if root is None:
            return cards
        for i in range(root.rowCount()):
            card_item = root.child(i)
            if card_item is None: continue
            
            card_model = self.reconstruct_card_model(card_item)
            if card_model:
                cards.append(card_model.model_dump(exclude_none=True))
        return cards

    def reconstruct_card_model(self, card_item) -> CardModel:
        # Get base data via Model (using construct to avoid strict validation failure on partial data during edit)
        card_model_data = self.get_item_model_data(card_item, CardModel)
        
        # We need to construct a new model that includes children
        if not card_model_data:
             # Fallback
             raw_data = self.get_item_data(card_item)
             card_model_data = CardModel.construct(**raw_data)

        # We will build a clean dict to re-instantiate CardModel at the end,
        # or modify card_model_data if it's mutable (Models are usually mutable by default in Pydantic V1/V2 unless frozen)

        # However, it's safer to build lists and assign
        effects = []
        static_abilities = []
        reaction_abilities = []
        spell_side = None
        keywords = card_model_data.keywords.copy() if card_model_data.keywords else {}

        # Iterate children
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            role = child.data(ROLE_TYPE)
            
            if role == "EFFECT":
                effects.append(self._reconstruct_effect(child))
            elif role == "MODIFIER":
                mod = self.get_item_model_data(child, ModifierModel)
                if mod: static_abilities.append(mod)
            elif role == "REACTION_ABILITY":
                react = self.get_item_model_data(child, ReactionModel)
                if react: reaction_abilities.append(react)
            elif role == "SPELL_SIDE":
                spell_side = self.reconstruct_card_model(child)
            elif role == "KEYWORDS":
                k_data = child.data(ROLE_DATA)
                if k_data: keywords.update(k_data)

        # Update lists on the model instance (or copy)
        card_model_data.effects = effects
        card_model_data.static_abilities = static_abilities
        card_model_data.reaction_abilities = reaction_abilities
        card_model_data.spell_side = spell_side
        card_model_data.keywords = keywords

        # Legacy keyword inference
        self._inject_keyword_logic_to_model(card_model_data)

        return card_model_data

    def _reconstruct_effect(self, eff_item) -> EffectModel:
        eff_model = self.get_item_model_data(eff_item, EffectModel)
        if not eff_model:
            raw = self.get_item_data(eff_item)
            eff_model = EffectModel.construct(**raw)

        commands = []
        for i in range(eff_item.rowCount()):
            child = eff_item.child(i)
            role = child.data(ROLE_TYPE)
            if role == "COMMAND":
                commands.append(self._reconstruct_command(child))
            # ACTION role removed: migration happens at load time.
            # If ACTION exists, it's dead data or needs manual conversion in UI (which is handled by convert tools)
            # We ignore it here to enforce "Clean Model" policy.

        eff_model.commands = commands
        return eff_model

    def _reconstruct_command(self, cmd_item) -> CommandModel:
        cmd_model = self.get_item_model_data(cmd_item, CommandModel)
        if not cmd_model:
            raw = self.get_item_data(cmd_item)
            cmd_model = CommandModel.construct(**raw)

        if_true_cmds = []
        if_false_cmds = []
        options_list = []

        for i in range(cmd_item.rowCount()):
            child = cmd_item.child(i)
            role = child.data(ROLE_TYPE)

            if role == "CMD_BRANCH_TRUE":
                for j in range(child.rowCount()):
                    if_true_cmds.append(self._reconstruct_command(child.child(j)))
            elif role == "CMD_BRANCH_FALSE":
                 for j in range(child.rowCount()):
                    if_false_cmds.append(self._reconstruct_command(child.child(j)))
            elif role == "OPTION":
                opt_cmds = []
                for j in range(child.rowCount()):
                     opt_cmds.append(self._reconstruct_command(child.child(j)))
                options_list.append(opt_cmds)

        cmd_model.if_true = if_true_cmds
        cmd_model.if_false = if_false_cmds
        cmd_model.options = options_list

        return cmd_model

    def _inject_keyword_logic_to_model(self, card_model: CardModel):
        # Update keywords based on commands (e.g. Revolution Change)
        # This modifies the model in-place
        for eff in card_model.effects:
            for cmd in eff.commands:
                if cmd.mutation_kind == 'REVOLUTION_CHANGE' or cmd.type == 'REVOLUTION_CHANGE':
                    card_model.keywords['revolution_change'] = True

    # --- Item Creation Wrappers ---

    def _create_card_item(self, model: CardModel):
        item = ModelAwareItem(f"{model.id} - {model.name}", model, "CARD")

        kw_item = QStandardItem(tr("Keywords"))
        kw_item.setData("KEYWORDS", ROLE_TYPE)
        kw_item.setData(model.keywords, ROLE_DATA)
        kw_item.setEditable(False)
        item.appendRow(kw_item)
        return item

    def _create_effect_item(self, model: EffectModel):
        return ModelAwareItem(f"{tr('Effect')}: {tr(model.trigger)}", model, "EFFECT")

    def _create_spell_side_item(self, model: CardModel):
        return ModelAwareItem(f"{tr('Spell Side')}: {model.name}", model, "SPELL_SIDE")

    def _create_modifier_item(self, model: ModifierModel):
        return ModelAwareItem(f"{tr('Static')}: {tr(model.type)}", model, "MODIFIER")

    def _create_reaction_item(self, model: ReactionModel):
        return ModelAwareItem(f"{tr('Reaction')}: {tr(model.type)}", model, "REACTION_ABILITY")

    def create_command_item(self, model_or_dict):
        if isinstance(model_or_dict, dict):
            model = CommandModel(**model_or_dict)
        else:
            model = model_or_dict

        label = f"{tr('Action')}: {tr(model.type)}"
        item = ModelAwareItem(label, model, "COMMAND")

        # Recursive rendering for branches/options
        if model.if_true:
            branch = QStandardItem(tr("If True"))
            branch.setData("CMD_BRANCH_TRUE", ROLE_TYPE)
            item.appendRow(branch)
            for cmd in model.if_true:
                branch.appendRow(self.create_command_item(cmd))

        if model.if_false:
            branch = QStandardItem(tr("If False"))
            branch.setData("CMD_BRANCH_FALSE", ROLE_TYPE)
            item.appendRow(branch)
            for cmd in model.if_false:
                branch.appendRow(self.create_command_item(cmd))

        if model.options:
            for idx, opt_cmds in enumerate(model.options):
                opt_item = QStandardItem(f"{tr('Option')} {idx+1}")
                opt_item.setData("OPTION", ROLE_TYPE)
                item.appendRow(opt_item)
                for cmd in opt_cmds:
                    opt_item.appendRow(self.create_command_item(cmd))

        return item

    # --- Utility ---

    def add_new_card(self):
        new_id = self._generate_new_id()
        model = CardModel(id=new_id, name="New Card")
        item = self._create_card_item(model)
        self.model.appendRow(item)
        return item

    def add_spell_side_item(self, card_item):
        """Create and attach a spell-side child to a card if absent."""
        card_item = self._ensure_item(card_item)
        if not card_item:
            return None

        # If already present, return existing
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child and child.data(ROLE_TYPE) == "SPELL_SIDE":
                return child

        card_data = self.get_item_model_data(card_item, CardModel)
        # fallback if not valid model
        if not card_data:
             card_data_dict = self.get_item_data(card_item)
             card_data = CardModel.construct(**card_data_dict)

        base_name = card_data.name
        civs = card_data.civilizations
        races = card_data.races
        cost = card_data.cost

        spell_model = CardModel(
            id=card_data.id,
            name=f"{base_name} (Spell Side)",
            type="SPELL",
            civilizations=civs,
            races=races,
            cost=cost,
            power=0,
            keywords={},
            effects=[],
            static_abilities=[],
            reaction_abilities=[]
        )

        child = self._create_spell_side_item(spell_model)
        card_item.appendRow(child)
        return child

    def remove_spell_side_item(self, card_item):
        card_item = self._ensure_item(card_item)
        if not card_item:
            return
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child and child.data(ROLE_TYPE) == "SPELL_SIDE":
                card_item.removeRow(i)
                return

    def _generate_new_id(self):
        # Simple max ID finder
        max_id = 0
        root = self.model.invisibleRootItem()
        if not root: return 1
        for i in range(root.rowCount()):
            c = root.child(i)
            if c:
                d = c.data(ROLE_DATA)
                if d and 'id' in d:
                    try:
                        cid = int(d['id'])
                        if cid > max_id: max_id = cid
                    except: pass
        return max_id + 1

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
        return model

    def create_default_static_data(self):
        """Create default data for a static ability."""
        model = ModifierModel(
            type="COST_MODIFIER",
            value=0,
            scope="ALL"
        )
        return model

    def add_child_item(self, parent_index, item_type, data, label):
        """Add a child item to a parent (e.g., effect to card, command to effect)."""
        from PyQt6.QtGui import QStandardItem
        from PyQt6.QtCore import QModelIndex
        
        if isinstance(parent_index, QModelIndex):
            parent_item = self.model.itemFromIndex(parent_index)
        else:
            parent_item = parent_index
        
        if not parent_item:
            return None
        
        # Determine model class from type if data is not already a model
        model_obj = data
        if isinstance(data, dict):
            if item_type == "EFFECT": model_obj = EffectModel(**data)
            elif item_type == "MODIFIER": model_obj = ModifierModel(**data)
            elif item_type == "REACTION_ABILITY": model_obj = ReactionModel(**data)
            elif item_type == "COMMAND": model_obj = CommandModel(**data)
            elif item_type == "CARD": model_obj = CardModel(**data)

        # Use ModelAwareItem
        # If model_obj is still a dict (unknown type), ModelAwareItem handles it as dict
        if isinstance(model_obj, BaseModel):
            new_item = ModelAwareItem(label, model_obj, item_type)
        else:
            new_item = QStandardItem(label)
            new_item.setData(item_type, ROLE_TYPE)
            new_item.setData(data, ROLE_DATA)

        parent_item.appendRow(new_item)
        return new_item

    def add_reaction(self, parent_index):
        """Add a reaction ability to a card."""
        model = ReactionModel(
            type="NINJA_STRIKE",
            cost=None,
            zone=None
        )
        label = f"{tr('Reaction')}: {model.type}"
        return self.add_child_item(parent_index, "REACTION_ABILITY", model, label)

    def create_default_command_data(self, cmd_data=None):
        """Create or copy default command data."""
        if cmd_data:
            if isinstance(cmd_data, BaseModel):
                return cmd_data.model_copy(deep=True)
            return copy.deepcopy(cmd_data)
        model = CommandModel(
            type="DRAW",
            amount=1
        )
        return model

    def format_command_label(self, cmd_data):
        """Format a label for a command item."""
        if isinstance(cmd_data, CommandModel):
            cmd_type = cmd_data.type
        else:
            cmd_type = cmd_data.get('type', 'UNKNOWN')
        return f"{tr('Action')}: {tr(cmd_type)}"

    def _sync_editor_warnings(self, card_item):
        pass # Placeholder for validation logic

    def convert_action_tree_to_command(self, action_item):
        # Backward compatibility for legacy Action items
        # Just grab data and assume it can be coerced to CommandModel
        act_data = self.get_item_data(action_item)
        # Use existing converter tool if needed
        try:
             cmd = ActionConverter.convert(act_data)
             return CommandModel(**cmd)
        except:
             # Fallback
             return CommandModel(type="NONE", str_param="Conversion Failed")

    def batch_convert_actions_recursive(self, item):
         # Stub
         return 0, 0

    # --- Template-driven Logic Insertion ---

    def apply_template_by_key(self, card_item, template_key, display_label=None):
        """
        Generic helper to apply a logic template to a card item.
        Replaces specific methods like add_revolution_change_logic.
        """
        if display_label is None:
             display_label = tr(template_key)

        card_data = self.get_item_data(card_item)
        # Prepare context for substitution
        context = {
            'civilizations': card_data.get('civilizations', ["FIRE"]),
            'races': card_data.get('races', [])
        }

        # Apply Template
        data, keywords_update, meta = self.template_manager.apply_template(template_key, context)

        if not data:
            print(f"Error: Template '{template_key}' not found or empty.")
            return None

        # Check specific requirements if needed (e.g. Mega Last Burst needs Spell Side)
        if template_key == "MEGA_LAST_BURST":
             has_spell_side = False
             for i in range(card_item.rowCount()):
                 child = card_item.child(i)
                 if child and child.data(ROLE_TYPE) == "SPELL_SIDE":
                     has_spell_side = True
                     break
             if not has_spell_side:
                 self.add_spell_side_item(card_item)

        # Create Model from Data
        # Assume root is EFFECT for now, based on meta['root_type']
        item = None
        if meta['root_type'] == 'EFFECT':
            model = EffectModel(**data)
            item = self._create_effect_item(model)
            item.setText(f"{tr('Effect')}: {display_label}") # Override label
            self._load_effect_children(item, model)
        else:
            # Fallback or other types if needed
            return None

        card_item.appendRow(item)

        # Update Keywords if needed
        if keywords_update:
            # Find keyword item
            kw_item = None
            for i in range(card_item.rowCount()):
                child = card_item.child(i)
                if child.data(ROLE_TYPE) == "KEYWORDS":
                    kw_item = child
                    break

            if kw_item:
                current_kws = kw_item.data(ROLE_DATA) or {}
                current_kws.update(keywords_update)
                kw_item.setData(current_kws, ROLE_DATA)

                # Also force update the underlying data model for preview
                try:
                    updated_model = self.reconstruct_card_model(card_item)
                    updated = updated_model.model_dump(by_alias=True) if hasattr(updated_model, 'model_dump') else updated_model.dict(by_alias=True)
                    if updated:
                        self.set_item_data(card_item, updated)
                except Exception:
                    pass

        return item

    def remove_logic_by_label(self, card_item, label_substring):
        for i in reversed(range(card_item.rowCount())):
             child = card_item.child(i)
             if label_substring in child.text():
                 card_item.removeRow(i)
                 # Force update preview
                 try:
                    updated_model = self.reconstruct_card_model(card_item)
                    updated = updated_model.model_dump(by_alias=True) if hasattr(updated_model, 'model_dump') else updated_model.dict(by_alias=True)
                    if updated:
                        self.set_item_data(card_item, updated)
                 except Exception:
                    pass
                 return

    def add_option_slots(self, parent_item, count):
        """
        Adds specified number of option slots to a COMMAND item.
        Ensures existing options are preserved if possible, or added if missing.
        Trims excess options.
        """
        if not parent_item: return

        current_count = 0
        options_to_remove = []

        # Scan current children
        for i in range(parent_item.rowCount()):
            child = parent_item.child(i)
            if child.data(ROLE_TYPE) == "OPTION":
                current_count += 1
                if current_count > count:
                    options_to_remove.append(i)

        # Remove excess (in reverse order)
        for i in reversed(options_to_remove):
            parent_item.removeRow(i)

        # Add missing
        for i in range(current_count, count):
            opt_item = QStandardItem(f"{tr('Option')} {i+1}")
            opt_item.setData("OPTION", ROLE_TYPE)
            parent_item.appendRow(opt_item)
