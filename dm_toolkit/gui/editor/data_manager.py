# -*- coding: utf-8 -*-
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QModelIndex
from dm_toolkit.gui.localization import tr
import uuid
import json
import os
import copy
from dm_toolkit.types import JSON
from dm_toolkit.gui.editor import normalize
from dm_toolkit.gui.editor.command_model import CommandDef, WarningCommand
from dm_toolkit.gui.editor.templates import LogicTemplateManager
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel, ReactionModel
from pydantic import BaseModel

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
        Returns data as a dictionary (for compatibility).
        Internally, it might extract from a Pydantic model stored in the item.
        """
        item = self._ensure_item(index_or_item)
        if item:
            data = item.data(ROLE_DATA)
            if isinstance(data, BaseModel):
                return data.model_dump(by_alias=True, exclude_none=True)
            return data or {}
        return {}

    def get_item_model(self, index_or_item):
        """
        Returns the raw data object stored in the item (preferably a Pydantic model).
        """
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_DATA)
        return None

    def set_item_data(self, index_or_item, data):
        """
        Sets data to the item.
        If 'data' is a Pydantic model, stores it directly.
        If 'data' is a dict, attempts to wrap it in a model if the type is known.
        """
        item = self._ensure_item(index_or_item)
        if item:
            # If it's already a model, store directly
            if isinstance(data, BaseModel):
                item.setData(data, ROLE_DATA)
            else:
                # If dict, try to keep it as dict unless we know better
                # Ideally caller should pass Model
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

    def _ensure_item(self, index_or_item):
        if isinstance(index_or_item, QModelIndex):
            return self.model.itemFromIndex(index_or_item)
        return index_or_item

    def load_data(self, cards_data):
        self.model.clear()
        self.model.setHorizontalHeaderLabels([tr("Logic Tree")])

        for card_raw in cards_data:
            # Validate/Convert using Pydantic
            try:
                # Handle legacy 'triggers' vs 'effects' normalization before model validation if needed
                if 'triggers' in card_raw:
                    card_raw['effects'] = card_raw.pop('triggers')

                # Legacy lifting of actions to commands removed as migration is complete.
                # All incoming data is expected to have 'commands'.

                card_model = CardModel(**card_raw)
            except Exception as e:
                print(f"Model validation failed for card {card_raw.get('id')}: {e}")
                # Fallback to raw dict if validation fails, but try to structure it
                # We try to make a model even if invalid, or fallback to dict
                try:
                    card_model = CardModel.construct(**card_raw)
                except:
                    card_model = None

            if card_model:
                card_item = self._create_card_item(card_model)

                # Effects
                for effect in card_model.effects:
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
        # Get base data
        raw_data = self.get_item_data(card_item) # Resolves to dict
        card_data = copy.deepcopy(raw_data)
        
        # Override lists
        card_data['effects'] = []
        card_data['static_abilities'] = []
        card_data['reaction_abilities'] = []

        # Iterate children
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            role = child.data(ROLE_TYPE)
            
            if role == "EFFECT":
                card_data['effects'].append(self._reconstruct_effect(child))
            elif role == "MODIFIER":
                card_data['static_abilities'].append(self.get_item_data(child))
            elif role == "REACTION_ABILITY":
                card_data['reaction_abilities'].append(self.get_item_data(child))
            elif role == "SPELL_SIDE":
                card_data['spell_side'] = self.reconstruct_card_model(child) # Recursively reconstruct
            elif role == "KEYWORDS":
                card_data['keywords'] = child.data(ROLE_DATA)

        # Legacy keyword inference (Revolution Change, etc.)
        self._inject_keyword_logic(card_data)

        return CardModel(**card_data)

    def _reconstruct_effect(self, eff_item) -> EffectModel:
        raw = self.get_item_data(eff_item)
        eff_data = copy.deepcopy(raw)
        eff_data['commands'] = []

        for i in range(eff_item.rowCount()):
            child = eff_item.child(i)
            if child.data(ROLE_TYPE) == "COMMAND":
                eff_data['commands'].append(self._reconstruct_command(child))
            # Legacy ACTION items are no longer supported/reconstructed

        return EffectModel(**eff_data)

    def _reconstruct_command(self, cmd_item) -> CommandModel:
        raw = self.get_item_data(cmd_item)
        cmd_data = copy.deepcopy(raw)

        # Handle recursive structures (if_true, if_false, options)
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

        if if_true_cmds: cmd_data['if_true'] = if_true_cmds
        if if_false_cmds: cmd_data['if_false'] = if_false_cmds
        if options_list: cmd_data['options'] = options_list

        return CommandModel(**cmd_data)

    def _inject_keyword_logic(self, card_data):
        keywords = card_data.get('keywords', {})
        for eff in card_data['effects']:
            # eff is EffectModel now
            cmds = eff.commands
            for cmd in cmds:
                # cmd is CommandModel
                if cmd.type == 'REVOLUTION_CHANGE':
                    keywords['revolution_change'] = True
        card_data['keywords'] = keywords

    # --- Item Creation Wrappers ---

    def _create_card_item(self, model: CardModel):
        item = QStandardItem(f"{model.id} - {model.name}")
        item.setData("CARD", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly

        kw_item = QStandardItem(tr("Keywords"))
        kw_item.setData("KEYWORDS", ROLE_TYPE)
        kw_item.setData(model.keywords, ROLE_DATA) # Dict
        kw_item.setEditable(False)
        item.appendRow(kw_item)
        return item

    def _create_effect_item(self, model: EffectModel):
        item = QStandardItem(f"{tr('Effect')}: {tr(model.trigger)}")
        item.setData("EFFECT", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly
        return item

    def _create_spell_side_item(self, model: CardModel):
        item = QStandardItem(f"{tr('Spell Side')}: {model.name}")
        item.setData("SPELL_SIDE", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly
        return item

    def _create_modifier_item(self, model: ModifierModel):
        item = QStandardItem(f"{tr('Static')}: {tr(model.type)}")
        item.setData("MODIFIER", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly
        return item

    def _create_reaction_item(self, model: ReactionModel):
        item = QStandardItem(f"{tr('Reaction')}: {tr(model.type)}")
        item.setData("REACTION_ABILITY", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly
        return item

    def create_command_item(self, model_or_dict):
        if isinstance(model_or_dict, dict):
            model = CommandModel(**model_or_dict)
        else:
            model = model_or_dict

        label = f"{tr('Action')}: {tr(model.type)}"
        item = QStandardItem(label)
        item.setData("COMMAND", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly

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

        card_data = self.get_item_data(card_item)
        base_name = card_data.get('name', "New Card")
        civs = card_data.get('civilizations', ["FIRE"])
        races = card_data.get('races', [])
        cost = card_data.get('cost', 1)

        spell_model = CardModel(
            id=card_data.get('id', 0),
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
                d = self.get_item_data(c)
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
        return model.model_dump()

    def create_default_static_data(self):
        """Create default data for a static ability."""
        model = ModifierModel(
            type="COST_MODIFIER",
            value=0,
            scope="ALL"
        )
        return model.model_dump()

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

    def _sync_editor_warnings(self, card_item):
        pass # Placeholder for validation logic

    def batch_convert_actions_recursive(self, item):
         # Stub - no-op as conversion is disabled/removed
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
                    # we must write back as Model now
                    self.set_item_data(card_item, updated_model)
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
                    self.set_item_data(card_item, updated_model)
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
