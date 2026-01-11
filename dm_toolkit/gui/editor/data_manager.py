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
from dm_toolkit.gui.editor.action_converter import ActionConverter, convert_action_to_objs
from dm_toolkit.gui.editor.command_model import CommandDef, WarningCommand
from dm_toolkit.gui.editor.templates import LogicTemplateManager
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel, ReactionModel

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
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_DATA) or {}
        return {}

    def set_item_data(self, index_or_item, data):
        item = self._ensure_item(index_or_item)
        if item:
            if hasattr(data, 'model_dump'): # Pydantic v2
                data = data.model_dump(by_alias=True)
            elif hasattr(data, 'dict'): # Pydantic v1
                data = data.dict(by_alias=True)
            item.setData(data, ROLE_DATA)

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

                # Basic cleaning of legacy command structures inside effects
                if 'effects' in card_raw:
                    for eff in card_raw['effects']:
                        self._lift_actions_to_commands(eff)

                card_model = CardModel(**card_raw)
            except Exception as e:
                print(f"Model validation failed for card {card_raw.get('id')}: {e}")
                # Fallback to raw dict if validation fails, but try to structure it
                card_model = CardModel.construct(**card_raw)

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

    def _lift_actions_to_commands(self, effect_data):
        # Helper to convert legacy "actions" list to "commands" if present in raw dict
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
        # Get base data
        raw_data = card_item.data(ROLE_DATA)
        # Deep copy to avoid mutating the item data directly during reconstruction
        card_data = copy.deepcopy(raw_data) if raw_data else {}
        
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
                card_data['static_abilities'].append(child.data(ROLE_DATA))
            elif role == "REACTION_ABILITY":
                card_data['reaction_abilities'].append(child.data(ROLE_DATA))
            elif role == "SPELL_SIDE":
                card_data['spell_side'] = self.reconstruct_card_model(child) # Recursively reconstruct
            elif role == "KEYWORDS":
                card_data['keywords'] = child.data(ROLE_DATA)

        # Legacy keyword inference (Revolution Change, etc.)
        self._inject_keyword_logic(card_data)

        return CardModel(**card_data)

    def _reconstruct_effect(self, eff_item) -> EffectModel:
        raw = eff_item.data(ROLE_DATA)
        eff_data = copy.deepcopy(raw)
        eff_data['commands'] = []

        for i in range(eff_item.rowCount()):
            child = eff_item.child(i)
            if child.data(ROLE_TYPE) == "COMMAND":
                eff_data['commands'].append(self._reconstruct_command(child))
            # Handle legacy ACTION conversion on the fly if still present
            elif child.data(ROLE_TYPE) == "ACTION":
                 cmd = self.convert_action_tree_to_command(child)
                 eff_data['commands'].append(cmd)

        return EffectModel(**eff_data)

    def _reconstruct_command(self, cmd_item) -> CommandModel:
        raw = cmd_item.data(ROLE_DATA)
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
        # Check commands for keywords like Revolution Change
        keywords = card_data.get('keywords', {})
        for eff in card_data['effects']:
            # eff is a EffectModel or dict at this point?
            # It's an EffectModel object if came from reconstruct, but we are modifying a dict before CardModel creation?
            # actually reconstruct_effect returns EffectModel.
            # We need to access it as model.
            cmds = eff.commands
            for cmd in cmds:
                if cmd.mutation_kind == 'REVOLUTION_CHANGE' or cmd.type == 'REVOLUTION_CHANGE':
                    keywords['revolution_change'] = True
                    # In a full impl, we'd extract the filter to card_data['revolution_change_condition']
        card_data['keywords'] = keywords

    # --- Item Creation Wrappers ---

    def _create_card_item(self, model: CardModel):
        item = QStandardItem(f"{model.id} - {model.name}")
        item.setData("CARD", ROLE_TYPE)
        item.setData(model.model_dump(), ROLE_DATA) # Store dict in Qt for compatibility

        kw_item = QStandardItem(tr("Keywords"))
        kw_item.setData("KEYWORDS", ROLE_TYPE)
        kw_item.setData(model.keywords, ROLE_DATA)
        kw_item.setEditable(False)
        item.appendRow(kw_item)
        return item

    def _create_effect_item(self, model: EffectModel):
        item = QStandardItem(f"{tr('Effect')}: {tr(model.trigger)}")
        item.setData("EFFECT", ROLE_TYPE)
        item.setData(model.model_dump(), ROLE_DATA)
        return item

    def _create_spell_side_item(self, model: CardModel):
        item = QStandardItem(f"{tr('Spell Side')}: {model.name}")
        item.setData("SPELL_SIDE", ROLE_TYPE)
        item.setData(model.model_dump(), ROLE_DATA)
        return item

    def _create_modifier_item(self, model: ModifierModel):
        item = QStandardItem(f"{tr('Static')}: {tr(model.type)}")
        item.setData("MODIFIER", ROLE_TYPE)
        item.setData(model.model_dump(), ROLE_DATA)
        return item

    def _create_reaction_item(self, model: ReactionModel):
        item = QStandardItem(f"{tr('Reaction')}: {tr(model.type)}")
        item.setData("REACTION_ABILITY", ROLE_TYPE)
        item.setData(model.model_dump(), ROLE_DATA)
        return item

    def create_command_item(self, model_or_dict):
        if isinstance(model_or_dict, dict):
            model = CommandModel(**model_or_dict)
        else:
            model = model_or_dict

        label = f"{tr('Action')}: {tr(model.type)}"
        item = QStandardItem(label)
        item.setData("COMMAND", ROLE_TYPE)
        item.setData(model.model_dump(), ROLE_DATA)

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
