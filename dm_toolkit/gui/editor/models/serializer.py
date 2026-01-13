# -*- coding: utf-8 -*-
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
from dm_toolkit.gui.i18n import tr
import copy
from pydantic import BaseModel

from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel, ReactionModel
from dm_toolkit.gui.editor.action_converter import ActionConverter, convert_action_to_objs

class ModelSerializer:
    """
    Responsible for converting between Pydantic models and QStandardItems,
    and serializing/deserializing the full card list.
    """

    def __init__(self):
        pass

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

    def get_item_model_obj(self, index_or_item):
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
        """
        item = self._ensure_item(index_or_item)
        if item:
            item.setData(data, ROLE_DATA)

    def _ensure_item(self, index_or_item):
        from PyQt6.QtCore import QModelIndex
        if isinstance(index_or_item, QModelIndex):
            if index_or_item.model():
                return index_or_item.model().itemFromIndex(index_or_item)
            return None
        return index_or_item

    def load_data(self, model: QStandardItemModel, cards_data: list):
        model.clear()
        model.setHorizontalHeaderLabels([tr("Logic Tree")])

        for card_raw in cards_data:
            # Validate/Convert using Pydantic
            try:
                # Handle legacy 'triggers' vs 'effects' normalization
                if 'triggers' in card_raw:
                    card_raw['effects'] = card_raw.pop('triggers')

                # Basic cleaning of legacy command structures inside effects
                if 'effects' in card_raw:
                    for eff in card_raw['effects']:
                        self._lift_actions_to_commands(eff)

                card_model = CardModel(**card_raw)
            except Exception as e:
                print(f"Model validation failed for card {card_raw.get('id')}: {e}")
                try:
                    card_model = CardModel.construct(**card_raw)
                except:
                    card_model = None

            if card_model:
                card_item = self.create_card_item(card_model)

                # Effects
                for effect in card_model.effects:
                    eff_item = self.create_effect_item(effect)
                    self._load_effect_children(eff_item, effect)
                    card_item.appendRow(eff_item)

                # Static Abilities
                for modifier in card_model.static_abilities:
                    mod_item = self.create_modifier_item(modifier)
                    card_item.appendRow(mod_item)

                # Reaction Abilities
                for reaction in card_model.reaction_abilities:
                    ra_item = self.create_reaction_item(reaction)
                    card_item.appendRow(ra_item)

                # Spell Side
                if card_model.spell_side:
                    spell_item = self.create_spell_side_item(card_model.spell_side)
                    for effect in card_model.spell_side.effects:
                        eff_item = self.create_effect_item(effect)
                        self._load_effect_children(eff_item, effect)
                        spell_item.appendRow(eff_item)
                    for modifier in card_model.spell_side.static_abilities:
                        mod_item = self.create_modifier_item(modifier)
                        spell_item.appendRow(mod_item)
                    card_item.appendRow(spell_item)

                model.appendRow(card_item)

    def get_full_data(self, model: QStandardItemModel):
        cards = []
        root = model.invisibleRootItem()
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

        # Legacy keyword inference
        # To avoid circular dependency, we import here or use a hook?
        # For now, we import the service helper inside the method.
        try:
            from dm_toolkit.gui.editor.services.feature_service import EditorFeatureService
            EditorFeatureService.inject_keyword_logic(card_data)
        except ImportError:
            pass

        return CardModel(**card_data)

    def _reconstruct_effect(self, eff_item) -> EffectModel:
        raw = self.get_item_data(eff_item)
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

    def convert_action_tree_to_command(self, action_item):
        act_data = self.get_item_data(action_item)
        try:
             cmd = ActionConverter.convert(act_data)
             return CommandModel(**cmd)
        except:
             return CommandModel(type="NONE", str_param="Conversion Failed")

    # --- Item Creation Methods ---

    def create_card_item(self, model: CardModel):
        item = QStandardItem(f"{model.id} - {model.name}")
        item.setData("CARD", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly

        kw_item = QStandardItem(tr("Keywords"))
        kw_item.setData("KEYWORDS", ROLE_TYPE)
        kw_item.setData(model.keywords, ROLE_DATA) # Dict
        kw_item.setEditable(False)
        item.appendRow(kw_item)
        return item

    def create_effect_item(self, model: EffectModel):
        item = QStandardItem(f"{tr('Effect')}: {tr(model.trigger)}")
        item.setData("EFFECT", ROLE_TYPE)
        item.setData(model, ROLE_DATA)
        return item

    def create_spell_side_item(self, model: CardModel):
        item = QStandardItem(f"{tr('Spell Side')}: {model.name}")
        item.setData("SPELL_SIDE", ROLE_TYPE)
        item.setData(model, ROLE_DATA)
        return item

    def create_modifier_item(self, model: ModifierModel):
        item = QStandardItem(f"{tr('Static')}: {tr(model.type)}")
        item.setData("MODIFIER", ROLE_TYPE)
        item.setData(model, ROLE_DATA)
        return item

    def create_reaction_item(self, model: ReactionModel):
        item = QStandardItem(f"{tr('Reaction')}: {tr(model.type)}")
        item.setData("REACTION_ABILITY", ROLE_TYPE)
        item.setData(model, ROLE_DATA)
        return item

    def create_command_item(self, model_or_dict):
        if isinstance(model_or_dict, dict):
            model = CommandModel(**model_or_dict)
        else:
            model = model_or_dict

        label = f"{tr('Action')}: {tr(model.type)}"
        item = QStandardItem(label)
        item.setData("COMMAND", ROLE_TYPE)
        item.setData(model, ROLE_DATA)

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

    def add_child_item(self, parent_item, item_type, data, label):
        """Add a child item to a parent."""
        if not parent_item:
            return None

        new_item = QStandardItem(label)
        new_item.setData(item_type, ROLE_TYPE)
        new_item.setData(data, ROLE_DATA)
        parent_item.appendRow(new_item)
        return new_item
