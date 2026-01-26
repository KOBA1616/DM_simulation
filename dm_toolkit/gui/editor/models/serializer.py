# -*- coding: utf-8 -*-
import copy
from typing import Optional, Any, List, Dict
from pydantic import BaseModel

from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel, ReactionModel
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel

class ModelSerializer:
    """
    Responsible for converting between Pydantic models and IEditorItems,
    and serializing/deserializing the full card list.
    """

    def __init__(self):
        pass

    def _ensure_item(self, item: Any) -> Optional[IEditorItem]:
        # Assume item is already IEditorItem or None
        if isinstance(item, IEditorItem):
            return item
        # If it's a Qt wrapper (QtEditorItem), it satisfies IEditorItem
        return None

    def get_item_data(self, item: IEditorItem) -> Dict[str, Any]:
        """
        Returns data as a dictionary (for compatibility).
        """
        if item:
            data = item.data(ROLE_DATA)
            if isinstance(data, BaseModel):
                return data.model_dump(by_alias=True, exclude_none=True)
            return data or {}
        return {}

    def get_item_model_obj(self, item: IEditorItem) -> Any:
        """
        Returns the raw data object stored in the item (preferably a Pydantic model).
        """
        if item:
            return item.data(ROLE_DATA)
        return None

    def set_item_data(self, item: IEditorItem, data: Any) -> None:
        """
        Sets data to the item.
        If 'data' is a Pydantic model, stores it directly.
        """
        if item:
            item.setData(data, ROLE_DATA)

    def load_data(self, model: IEditorModel, cards_data: list):
        model.clear()
        model.setHorizontalHeaderLabels([tr("Logic Tree")])

        for card_raw in cards_data:
            # Validate/Convert using Pydantic
            try:
                # Handle legacy 'triggers' vs 'effects' normalization
                if 'triggers' in card_raw:
                    card_raw['effects'] = card_raw.pop('triggers')

                card_model = CardModel(**card_raw)
            except Exception as e:
                print(f"Model validation failed for card {card_raw.get('id')}: {e}")
                try:
                    card_model = CardModel.construct(**card_raw)
                except:
                    card_model = None

            if card_model:
                card_item = self.create_card_item(card_model, model)

                # Effects
                for effect in card_model.effects:
                    eff_item = self.create_effect_item(effect, model)
                    self._load_effect_children(eff_item, effect, model)
                    card_item.appendRow(eff_item)

                # Static Abilities
                for modifier in card_model.static_abilities:
                    mod_item = self.create_modifier_item(modifier, model)
                    card_item.appendRow(mod_item)

                # Reaction Abilities
                for reaction in card_model.reaction_abilities:
                    ra_item = self.create_reaction_item(reaction, model)
                    card_item.appendRow(ra_item)

                # Spell Side
                if card_model.spell_side:
                    spell_item = self.create_spell_side_item(card_model.spell_side, model)
                    for effect in card_model.spell_side.effects:
                        eff_item = self.create_effect_item(effect, model)
                        self._load_effect_children(eff_item, effect, model)
                        spell_item.appendRow(eff_item)
                    for modifier in card_model.spell_side.static_abilities:
                        mod_item = self.create_modifier_item(modifier, model)
                        spell_item.appendRow(mod_item)
                    card_item.appendRow(spell_item)

                model.appendRow(card_item)

    def get_full_data(self, model: IEditorModel):
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

    def reconstruct_card_model(self, card_item: IEditorItem) -> CardModel:
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
            if not child: continue
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
        try:
            from dm_toolkit.gui.editor.services.feature_service import EditorFeatureService
            EditorFeatureService.inject_keyword_logic(card_data)
        except ImportError:
            pass

        return CardModel(**card_data)

    def _reconstruct_effect(self, eff_item: IEditorItem) -> EffectModel:
        raw = self.get_item_data(eff_item)
        eff_data = copy.deepcopy(raw)
        eff_data['commands'] = []

        for i in range(eff_item.rowCount()):
            child = eff_item.child(i)
            if child and child.data(ROLE_TYPE) == "COMMAND":
                eff_data['commands'].append(self._reconstruct_command(child))

        return EffectModel(**eff_data)

    def _reconstruct_command(self, cmd_item: IEditorItem) -> CommandModel:
        raw = self.get_item_data(cmd_item)
        cmd_data = copy.deepcopy(raw)

        # Handle recursive structures (if_true, if_false, options)
        if_true_cmds = []
        if_false_cmds = []
        options_list = []

        for i in range(cmd_item.rowCount()):
            child = cmd_item.child(i)
            if not child: continue
            role = child.data(ROLE_TYPE)

            if role == "CMD_BRANCH_TRUE":
                for j in range(child.rowCount()):
                    c = child.child(j)
                    if c: if_true_cmds.append(self._reconstruct_command(c))
            elif role == "CMD_BRANCH_FALSE":
                 for j in range(child.rowCount()):
                    c = child.child(j)
                    if c: if_false_cmds.append(self._reconstruct_command(c))
            elif role == "OPTION":
                opt_cmds = []
                for j in range(child.rowCount()):
                     c = child.child(j)
                     if c: opt_cmds.append(self._reconstruct_command(c))
                options_list.append(opt_cmds)

        if if_true_cmds: cmd_data['if_true'] = if_true_cmds
        if if_false_cmds: cmd_data['if_false'] = if_false_cmds
        if options_list: cmd_data['options'] = options_list

        return CommandModel(**cmd_data)

    def _load_effect_children(self, eff_item: IEditorItem, effect_model: EffectModel, factory: IEditorModel):
        for command in effect_model.commands:
            cmd_item = self.create_command_item(command, factory)
            eff_item.appendRow(cmd_item)


    # --- Item Creation Methods ---

    def create_card_item(self, model: CardModel, factory: IEditorModel) -> IEditorItem:
        item = factory.create_item(f"{model.id} - {model.name}")
        item.setData("CARD", ROLE_TYPE)
        item.setData(model, ROLE_DATA) # Store Model directly

        kw_item = factory.create_item(tr("Keywords"))
        kw_item.setData("KEYWORDS", ROLE_TYPE)
        kw_item.setData(model.keywords, ROLE_DATA) # Dict
        kw_item.setEditable(False)
        item.appendRow(kw_item)
        return item

    def create_effect_item(self, model: EffectModel, factory: IEditorModel) -> IEditorItem:
        item = factory.create_item(f"{tr('Effect')}: {tr(model.trigger)}")
        item.setData("EFFECT", ROLE_TYPE)
        item.setData(model, ROLE_DATA)
        return item

    def create_spell_side_item(self, model: CardModel, factory: IEditorModel) -> IEditorItem:
        item = factory.create_item(f"{tr('Spell Side')}: {model.name}")
        item.setData("SPELL_SIDE", ROLE_TYPE)
        item.setData(model, ROLE_DATA)

        kw_item = factory.create_item(tr("Keywords"))
        kw_item.setData("KEYWORDS", ROLE_TYPE)
        kw_item.setData(model.keywords, ROLE_DATA) # Dict
        kw_item.setEditable(False)
        item.appendRow(kw_item)
        return item

    def create_modifier_item(self, model: ModifierModel, factory: IEditorModel) -> IEditorItem:
        item = factory.create_item(f"{tr('Static')}: {tr(model.type)}")
        item.setData("MODIFIER", ROLE_TYPE)
        item.setData(model, ROLE_DATA)
        return item

    def create_reaction_item(self, model: ReactionModel, factory: IEditorModel) -> IEditorItem:
        item = factory.create_item(f"{tr('Reaction')}: {tr(model.type)}")
        item.setData("REACTION_ABILITY", ROLE_TYPE)
        item.setData(model, ROLE_DATA)
        return item

    def create_command_item(self, model_or_dict, factory: IEditorModel) -> IEditorItem:
        if isinstance(model_or_dict, dict):
            model = CommandModel(**model_or_dict)
        else:
            model = model_or_dict

        label = f"{tr('Action')}: {tr(model.type)}"
        item = factory.create_item(label)
        item.setData("COMMAND", ROLE_TYPE)
        item.setData(model, ROLE_DATA)

        # Recursive rendering for branches/options
        if model.if_true:
            branch = factory.create_item(tr("If True"))
            branch.setData("CMD_BRANCH_TRUE", ROLE_TYPE)
            item.appendRow(branch)
            for cmd in model.if_true:
                branch.appendRow(self.create_command_item(cmd, factory))

        if model.if_false:
            branch = factory.create_item(tr("If False"))
            branch.setData("CMD_BRANCH_FALSE", ROLE_TYPE)
            item.appendRow(branch)
            for cmd in model.if_false:
                branch.appendRow(self.create_command_item(cmd, factory))

        if model.options:
            for idx, opt_cmds in enumerate(model.options):
                opt_item = factory.create_item(f"{tr('Option')} {idx+1}")
                opt_item.setData("OPTION", ROLE_TYPE)
                item.appendRow(opt_item)
                for cmd in opt_cmds:
                    opt_item.appendRow(self.create_command_item(cmd, factory))

        return item

    def add_child_item(self, parent_item: IEditorItem, item_type: str, data: Any, label: str, factory: IEditorModel) -> Optional[IEditorItem]:
        """Add a child item to a parent."""
        if not parent_item:
            return None

        new_item = factory.create_item(label)
        new_item.setData(item_type, ROLE_TYPE)
        new_item.setData(data, ROLE_DATA)
        parent_item.appendRow(new_item)
        return new_item
