# dm_toolkit/gui/editor/models/serializer.py
# -*- coding: utf-8 -*-
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel
from dm_toolkit.gui.i18n import tr
import copy
from pydantic import BaseModel

from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ModifierModel, ReactionModel

class ModelSerializer:
    """
    Responsible for converting between Pydantic models and IEditorItems,
    and serializing/deserializing the full card list.
    """

    def __init__(self):
        pass

    def _ensure_item(self, item) -> IEditorItem:
        if isinstance(item, IEditorItem):
            return item
        return None

    def get_item_data(self, item):
        """
        Returns data as a dictionary (for compatibility).
        Internally, it might extract from a Pydantic model stored in the item.
        """
        item = self._ensure_item(item)
        if item:
            data = item.data(ROLE_DATA)
            if isinstance(data, BaseModel):
                return data.model_dump(by_alias=True, exclude_none=True)
            return data or {}
        return {}

    def get_item_model_obj(self, item):
        """
        Returns the raw data object stored in the item (preferably a Pydantic model).
        """
        item = self._ensure_item(item)
        if item:
            return item.data(ROLE_DATA)
        return None

    def set_item_data(self, item, data):
        """
        Sets data to the item.
        If 'data' is a Pydantic model, stores it directly.
        """
        item = self._ensure_item(item)
        if item:
            item.set_data(data, ROLE_DATA)

    def load_data(self, model: IEditorModel, cards_data: list):
        model.clear()
        model.set_headers([tr("Logic Tree")])

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
                card_item = self.create_card_item(model, card_model)

                # Effects
                for effect in card_model.effects:
                    eff_item = self.create_effect_item(model, effect)
                    self._load_effect_children(model, eff_item, effect)
                    card_item.append_row(eff_item)

                # Static Abilities
                for modifier in card_model.static_abilities:
                    mod_item = self.create_modifier_item(model, modifier)
                    card_item.append_row(mod_item)

                # Reaction Abilities
                for reaction in card_model.reaction_abilities:
                    ra_item = self.create_reaction_item(model, reaction)
                    card_item.append_row(ra_item)

                # Spell Side
                if card_model.spell_side:
                    spell_item = self.create_spell_side_item(model, card_model.spell_side)
                    for effect in card_model.spell_side.effects:
                        eff_item = self.create_effect_item(model, effect)
                        self._load_effect_children(model, eff_item, effect)
                        spell_item.append_row(eff_item)
                    for modifier in card_model.spell_side.static_abilities:
                        mod_item = self.create_modifier_item(model, modifier)
                        spell_item.append_row(mod_item)
                    card_item.append_row(spell_item)

                model.append_row(card_item)

    def get_full_data(self, model: IEditorModel):
        cards = []
        root = model.root_item()
        if root is None:
            return cards
        for i in range(root.row_count()):
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
        for i in range(card_item.row_count()):
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

        for i in range(eff_item.row_count()):
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

        for i in range(cmd_item.row_count()):
            child = cmd_item.child(i)
            if not child: continue
            role = child.data(ROLE_TYPE)

            if role == "CMD_BRANCH_TRUE":
                for j in range(child.row_count()):
                    if_true_cmds.append(self._reconstruct_command(child.child(j)))
            elif role == "CMD_BRANCH_FALSE":
                 for j in range(child.row_count()):
                    if_false_cmds.append(self._reconstruct_command(child.child(j)))
            elif role == "OPTION":
                opt_cmds = []
                for j in range(child.row_count()):
                     opt_cmds.append(self._reconstruct_command(child.child(j)))
                options_list.append(opt_cmds)

        if if_true_cmds: cmd_data['if_true'] = if_true_cmds
        if if_false_cmds: cmd_data['if_false'] = if_false_cmds
        if options_list: cmd_data['options'] = options_list

        return CommandModel(**cmd_data)

    def _load_effect_children(self, model: IEditorModel, eff_item: IEditorItem, effect_model: EffectModel):
        for command in effect_model.commands:
            cmd_item = self.create_command_item(model, command)
            eff_item.append_row(cmd_item)


    # --- Item Creation Methods ---

    def create_card_item(self, model: IEditorModel, card_model: CardModel) -> IEditorItem:
        item = model.create_item(f"{card_model.id} - {card_model.name}")
        item.set_data("CARD", ROLE_TYPE)
        item.set_data(card_model, ROLE_DATA) # Store Model directly

        kw_item = model.create_item(tr("Keywords"))
        kw_item.set_data("KEYWORDS", ROLE_TYPE)
        kw_item.set_data(card_model.keywords, ROLE_DATA) # Dict
        kw_item.set_editable(False)
        item.append_row(kw_item)
        return item

    def create_effect_item(self, model: IEditorModel, effect_model: EffectModel) -> IEditorItem:
        item = model.create_item(f"{tr('Effect')}: {tr(effect_model.trigger)}")
        item.set_data("EFFECT", ROLE_TYPE)
        item.set_data(effect_model, ROLE_DATA)
        return item

    def create_spell_side_item(self, model: IEditorModel, card_model: CardModel) -> IEditorItem:
        item = model.create_item(f"{tr('Spell Side')}: {card_model.name}")
        item.set_data("SPELL_SIDE", ROLE_TYPE)
        item.set_data(card_model, ROLE_DATA)

        kw_item = model.create_item(tr("Keywords"))
        kw_item.set_data("KEYWORDS", ROLE_TYPE)
        kw_item.set_data(card_model.keywords, ROLE_DATA) # Dict
        kw_item.set_editable(False)
        item.append_row(kw_item)
        return item

    def create_modifier_item(self, model: IEditorModel, modifier_model: ModifierModel) -> IEditorItem:
        item = model.create_item(f"{tr('Static')}: {tr(modifier_model.type)}")
        item.set_data("MODIFIER", ROLE_TYPE)
        item.set_data(modifier_model, ROLE_DATA)
        return item

    def create_reaction_item(self, model: IEditorModel, reaction_model: ReactionModel) -> IEditorItem:
        item = model.create_item(f"{tr('Reaction')}: {tr(reaction_model.type)}")
        item.set_data("REACTION_ABILITY", ROLE_TYPE)
        item.set_data(reaction_model, ROLE_DATA)
        return item

    def create_command_item(self, model: IEditorModel, model_or_dict) -> IEditorItem:
        if isinstance(model_or_dict, dict):
            cmd_model = CommandModel(**model_or_dict)
        else:
            cmd_model = model_or_dict

        label = f"{tr('Action')}: {tr(cmd_model.type)}"
        item = model.create_item(label)
        item.set_data("COMMAND", ROLE_TYPE)
        item.set_data(cmd_model, ROLE_DATA)

        # Recursive rendering for branches/options
        if cmd_model.if_true:
            branch = model.create_item(tr("If True"))
            branch.set_data("CMD_BRANCH_TRUE", ROLE_TYPE)
            item.append_row(branch)
            for cmd in cmd_model.if_true:
                branch.append_row(self.create_command_item(model, cmd))

        if cmd_model.if_false:
            branch = model.create_item(tr("If False"))
            branch.set_data("CMD_BRANCH_FALSE", ROLE_TYPE)
            item.append_row(branch)
            for cmd in cmd_model.if_false:
                branch.append_row(self.create_command_item(model, cmd))

        if cmd_model.options:
            for idx, opt_cmds in enumerate(cmd_model.options):
                opt_item = model.create_item(f"{tr('Option')} {idx+1}")
                opt_item.set_data("OPTION", ROLE_TYPE)
                item.append_row(opt_item)
                for cmd in opt_cmds:
                    opt_item.append_row(self.create_command_item(model, cmd))

        return item

    def add_child_item(self, model: IEditorModel, parent_item: IEditorItem, item_type, data, label) -> IEditorItem:
        """Add a child item to a parent."""
        if not parent_item:
            return None

        new_item = model.create_item(label)
        new_item.set_data(item_type, ROLE_TYPE)
        new_item.set_data(data, ROLE_DATA)
        parent_item.append_row(new_item)
        return new_item
