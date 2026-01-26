# -*- coding: utf-8 -*-
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ReactionModel
from dm_toolkit.gui.editor.templates import LogicTemplateManager
from dm_toolkit.editor.core.abstraction import IEditorModel, IEditorItem

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from dm_toolkit.gui.editor.models.serializer import ModelSerializer

class EditorFeatureService:
    """
    Handles domain-specific logic for the card editor, such as adding new cards,
    applying templates, and injecting business logic.
    """

    def __init__(self, model: IEditorModel, serializer: 'ModelSerializer'):
        self.model = model
        self.serializer = serializer
        self.template_manager = LogicTemplateManager.get_instance()

    def add_new_card(self):
        new_id = self._generate_new_id()
        model = CardModel(id=new_id, name="New Card")
        # serializer.create_card_item now takes model
        item = self.serializer.create_card_item(self.model, model)
        self.model.append_row(item)
        return item

    def _generate_new_id(self):
        # Simple max ID finder
        max_id = 0
        root = self.model.root_item()
        if not root: return 1
        for i in range(root.row_count()):
            c = root.child(i)
            if c:
                d = self.serializer.get_item_data(c)
                if d and 'id' in d:
                    try:
                        cid = int(d['id'])
                        if cid > max_id: max_id = cid
                    except: pass
        return max_id + 1

    def add_spell_side_item(self, card_item: IEditorItem):
        """Create and attach a spell-side child to a card if absent."""
        # card_item assumed to be IEditorItem. Caller ensures it.
        # Check if already present
        for i in range(card_item.row_count()):
            child = card_item.child(i)
            if child and child.data(ROLE_TYPE) == "SPELL_SIDE":
                return child

        card_data = self.serializer.get_item_data(card_item)
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

        child = self.serializer.create_spell_side_item(self.model, spell_model)
        card_item.append_row(child)
        return child

    def remove_spell_side_item(self, card_item: IEditorItem):
        if not card_item:
            return
        for i in range(card_item.row_count()):
            child = card_item.child(i)
            if child and child.data(ROLE_TYPE) == "SPELL_SIDE":
                card_item.remove_row(i)
                return

    def add_reaction(self, parent_item: IEditorItem):
        """Add a reaction ability to a card."""
        model = ReactionModel(
            type="NINJA_STRIKE",
            cost=None,
            zone=None
        )

        item = self.serializer.create_reaction_item(self.model, model)
        parent_item.append_row(item)
        return item

    def apply_template_by_key(self, card_item: IEditorItem, template_key, display_label=None):
        if display_label is None:
             display_label = tr(template_key)

        card_data = self.serializer.get_item_data(card_item)
        context = {
            'civilizations': card_data.get('civilizations', ["FIRE"]),
            'races': card_data.get('races', [])
        }

        data, keywords_update, meta = self.template_manager.apply_template(template_key, context)

        if not data:
            print(f"Error: Template '{template_key}' not found or empty.")
            return None

        if template_key == "MEGA_LAST_BURST":
             has_spell_side = False
             for i in range(card_item.row_count()):
                 child = card_item.child(i)
                 if child and child.data(ROLE_TYPE) == "SPELL_SIDE":
                     has_spell_side = True
                     break
             if not has_spell_side:
                 self.add_spell_side_item(card_item)

        item = None
        if meta['root_type'] == 'EFFECT':
            model = EffectModel(**data)
            item = self.serializer.create_effect_item(self.model, model)
            item.set_text(f"{tr('Effect')}: {display_label}")
            self.serializer._load_effect_children(self.model, item, model)
        else:
            return None

        card_item.append_row(item)

        if keywords_update:
            kw_item = None
            for i in range(card_item.row_count()):
                child = card_item.child(i)
                if child.data(ROLE_TYPE) == "KEYWORDS":
                    kw_item = child
                    break

            if kw_item:
                current_kws = kw_item.data(ROLE_DATA) or {}
                current_kws.update(keywords_update)
                kw_item.set_data(current_kws, ROLE_DATA)

                try:
                    updated_model = self.serializer.reconstruct_card_model(card_item)
                    self.serializer.set_item_data(card_item, updated_model)
                except Exception:
                    pass

        return item

    def remove_logic_by_label(self, card_item: IEditorItem, label_substring):
        for i in reversed(range(card_item.row_count())):
             child = card_item.child(i)
             if label_substring in child.text():
                 card_item.remove_row(i)
                 try:
                    updated_model = self.serializer.reconstruct_card_model(card_item)
                    self.serializer.set_item_data(card_item, updated_model)
                 except Exception:
                    pass
                 return

    def add_option_slots(self, parent_item: IEditorItem, count):
        if not parent_item: return

        current_count = 0
        options_to_remove = []

        for i in range(parent_item.row_count()):
            child = parent_item.child(i)
            if child.data(ROLE_TYPE) == "OPTION":
                current_count += 1
                if current_count > count:
                    options_to_remove.append(i)

        for i in reversed(options_to_remove):
            parent_item.remove_row(i)

        for i in range(current_count, count):
            opt_item = self.model.create_item(f"{tr('Option')} {i+1}")
            opt_item.set_data("OPTION", ROLE_TYPE)
            parent_item.append_row(opt_item)

    @staticmethod
    def inject_keyword_logic(card_data):
        keywords = card_data.get('keywords', {})
        effects = card_data.get('effects', [])

        for eff in effects:
            cmds = []
            if hasattr(eff, 'commands'):
                cmds = eff.commands
            elif isinstance(eff, dict):
                cmds = eff.get('commands', [])

            for cmd in cmds:
                ctype = getattr(cmd, 'type', None) or (cmd.get('type') if isinstance(cmd, dict) else None)
                if ctype == 'REVOLUTION_CHANGE':
                    keywords['revolution_change'] = True
                elif ctype == 'FRIEND_BURST':
                    keywords['friend_burst'] = True
                    target_filter = getattr(cmd, 'target_filter', None) or (cmd.get('target_filter') if isinstance(cmd, dict) else None)
                    if target_filter:
                        card_data['friend_burst_condition'] = target_filter

        card_data['keywords'] = keywords
