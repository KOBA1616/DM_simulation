# -*- coding: utf-8 -*-
from typing import TYPE_CHECKING, Any
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.consts import ROLE_TYPE, ROLE_DATA
from dm_toolkit.gui.editor.models import CardModel, EffectModel, CommandModel, ReactionModel
from dm_toolkit.gui.editor.templates import LogicTemplateManager
from dm_toolkit.editor.core.abstraction import IEditorItem, IEditorModel

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
        item = self.serializer.create_card_item(model, self.model)
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
                d = self.serializer.get_item_data(c)
                if d and 'id' in d:
                    try:
                        cid = int(d['id'])
                        if cid > max_id: max_id = cid
                    except: pass
        return max_id + 1

    def add_spell_side_item(self, card_item):
        """Create and attach a spell-side child to a card if absent."""
        card_item = self.serializer._ensure_item(card_item)
        if not card_item:
            return None

        # If already present, return existing
        for i in range(card_item.rowCount()):
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

        child = self.serializer.create_spell_side_item(spell_model, self.model)
        card_item.appendRow(child)
        return child

    def remove_spell_side_item(self, card_item):
        card_item = self.serializer._ensure_item(card_item)
        if not card_item:
            return
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child and child.data(ROLE_TYPE) == "SPELL_SIDE":
                card_item.removeRow(i)
                return

    def add_reaction(self, parent_index):
        """Add a reaction ability to a card."""
        model = ReactionModel(
            type="NINJA_STRIKE",
            cost=None,
            zone=None
        )
        label = f"{tr('Reaction')}: {model.type}"
        # We need parent_item
        if isinstance(parent_index, IEditorItem):
            parent_item = parent_index
        else:
            parent_item = self.model.itemFromIndex(parent_index)

        return self.serializer.add_child_item(parent_item, "REACTION_ABILITY", model, label, self.model)

    def apply_template_by_key(self, card_item, template_key, display_label=None):
        """
        Generic helper to apply a logic template to a card item.
        """
        card_item = self.serializer._ensure_item(card_item)
        if not card_item: return None

        if display_label is None:
             display_label = tr(template_key)

        card_data = self.serializer.get_item_data(card_item)
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

        # Check specific requirements if needed
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
        item = None
        if meta['root_type'] == 'EFFECT':
            model = EffectModel(**data)
            item = self.serializer.create_effect_item(model, self.model)
            item.setText(f"{tr('Effect')}: {display_label}") # Override label
            self.serializer._load_effect_children(item, model, self.model)
        else:
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

                # Force update underlying data model for preview
                try:
                    updated_model = self.serializer.reconstruct_card_model(card_item)
                    self.serializer.set_item_data(card_item, updated_model)
                except Exception:
                    pass

        return item

    def remove_logic_by_label(self, card_item, label_substring):
        card_item = self.serializer._ensure_item(card_item)
        if not card_item: return

        for i in reversed(range(card_item.rowCount())):
             child = card_item.child(i)
             if label_substring in child.text():
                 card_item.removeRow(i)
                 try:
                    updated_model = self.serializer.reconstruct_card_model(card_item)
                    self.serializer.set_item_data(card_item, updated_model)
                 except Exception:
                    pass
                 return

    def add_option_slots(self, parent_item, count):
        """
        Adds specified number of option slots to a COMMAND item.
        """
        parent_item = self.serializer._ensure_item(parent_item)
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

        # Remove excess
        for i in reversed(options_to_remove):
            parent_item.removeRow(i)

        # Add missing
        for i in range(current_count, count):
            opt_item = self.model.create_item(f"{tr('Option')} {i+1}")
            opt_item.setData("OPTION", ROLE_TYPE)
            parent_item.appendRow(opt_item)

    @staticmethod
    def inject_keyword_logic(card_data):
        keywords = card_data.get('keywords', {})
        effects = card_data.get('effects', [])

        for eff in effects:
            # Handle both dict and Model
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
