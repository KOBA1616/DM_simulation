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
from dm_toolkit.gui.editor.models import CardNode, CommandModel, BaseModel

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Handles loading, saving (reconstruction), and item creation (ID generation).
    Uses data models wrapper where possible.
    """

    def __init__(self, model: QStandardItemModel):
        self.model = model
        self._internal_cache = {}
        self.template_manager = LogicTemplateManager.get_instance()

    def get_item_type(self, index_or_item):
        """Safe accessor for item type (UserRole+1)."""
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_TYPE)
        return None

    def get_item_data(self, index_or_item):
        """Safe accessor for item data (UserRole+2). Returns empty dict if None."""
        item = self._ensure_item(index_or_item)
        if item:
            return item.data(ROLE_DATA) or {}
        return {}

    def set_item_data(self, index_or_item, data):
        """Safe setter for item data (UserRole+2)."""
        item = self._ensure_item(index_or_item)
        if item:
            # If data is a model instance, extract dict
            if hasattr(data, 'to_dict'):
                data = data.to_dict()
            item.setData(data, ROLE_DATA)

    def _ensure_item(self, index_or_item):
        if isinstance(index_or_item, QModelIndex):
            return self.model.itemFromIndex(index_or_item)
        return index_or_item

    def get_card_context_type(self, index_or_item):
        item = self._ensure_item(index_or_item)
        card_item = self._find_card_root(item)
        if card_item:
            cdata = self.get_item_data(card_item)
            return cdata.get('type', 'CREATURE')
        return 'CREATURE'

    def create_default_trigger_data(self):
        return {
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "commands": []
        }

    def create_default_static_data(self):
        return {
            "type": "COST_MODIFIER",
            "value": -1,
            "str_val": "",
            "scope": "ALL",
            "condition": {"type": "NONE"},
            "filter": {}
        }

    def create_default_reaction_data(self):
        return {
            "type": "NINJA_STRIKE",
            "cost": 4,
            "zone": "HAND",
            "condition": {
                "trigger_event": "ON_BLOCK_OR_ATTACK",
                "civilization_match": True,
                "mana_count_min": 0
            }
        }

    def create_default_command_data(self, template=None):
        if template:
            if 'data' in template:
                 import copy
                 data = copy.deepcopy(template['data'])
                 if 'uid' in data: del data['uid']
                 return data
            import copy
            data = copy.deepcopy(template)
            if 'uid' in data: del data['uid']
            return data

        return {
            "type": "TRANSITION",
            "target_group": "NONE",
            "to_zone": "HAND",
            "target_filter": {}
        }

    def validate_card_data(self, card_data: dict) -> list[str]:
        try:
            node = CardNode.from_json(card_data)
            return node.validate()
        except Exception as e:
            return [f"Model Validation Error: {str(e)}"]

    def update_effect_type(self, item, target_type):
        data = self.get_item_data(item)

        if target_type == "TRIGGERED":
            item.setData("EFFECT", ROLE_TYPE)
            trigger = data.get('trigger', 'NONE')
            item.setText(f"{tr('Effect')}: {tr(trigger)}")

        elif target_type == "STATIC":
            item.setData("MODIFIER", ROLE_TYPE)
            mtype = data.get('type', data.get('layer_type', 'NONE'))
            item.setText(f"{tr('Static')}: {tr(mtype)}")

    def _ensure_uid(self, obj: dict):
        if obj is None or not isinstance(obj, dict):
            return
        if 'uid' not in obj:
            obj['uid'] = str(uuid.uuid4())

    def _internalize_item(self, item):
        try:
            data = item.data(ROLE_DATA)
        except Exception:
            data = None
        internal = normalize.canonicalize(data)
        uid = None
        if isinstance(data, dict):
            uid = data.get('uid')
        if not uid:
            uid = str(uuid.uuid4())
        self._internal_cache[uid] = internal
        return internal

    def _lift_actions_to_commands(self, effect_data):
        try:
            legacy_actions = effect_data.get('actions', [])
            if legacy_actions:
                converted_cmds = []
                for act in list(legacy_actions):
                    try:
                        objs = convert_action_to_objs(act)
                        for o in objs:
                            if hasattr(o, 'to_dict'):
                                converted_cmds.append(o.to_dict())
                            elif isinstance(o, dict):
                                converted_cmds.append(o)
                            else:
                                converted_cmds.append({
                                    'type': 'NONE',
                                    'legacy_warning': True,
                                    'legacy_original_action': act
                                })
                    except Exception:
                        converted_cmds.append({
                            'type': 'NONE',
                            'legacy_warning': True,
                            'legacy_original_action': act
                        })
                effect_data['commands'] = effect_data.get('commands', []) + converted_cmds

            if 'commands' not in effect_data:
                effect_data['commands'] = []

            if 'actions' in effect_data:
                try:
                    del effect_data['actions']
                except Exception:
                    pass
        except Exception:
            pass

    def load_data(self, cards_data):
        self.model.clear()
        from dm_toolkit.gui.localization import tr
        self.model.setHorizontalHeaderLabels([tr("Logic Tree")])

        for card_idx, card in enumerate(cards_data):
            if 'triggers' in card:
                card['_meta_use_triggers'] = True

            card_item = self._create_card_item(card)

            triggers = card.get('triggers', [])
            if not triggers:
                triggers = card.get('effects', [])

            for eff_idx, effect in enumerate(triggers):
                self._lift_actions_to_commands(effect)
                eff_item = self._create_effect_item(effect)
                self._load_effect_children(eff_item, effect)
                card_item.appendRow(eff_item)

            for mod_idx, modifier in enumerate(card.get('static_abilities', [])):
                 mod_item = self._create_modifier_item(modifier)
                 card_item.appendRow(mod_item)

            for ra_idx, ra in enumerate(card.get('reaction_abilities', [])):
                ra_item = self._create_reaction_item(ra)
                if isinstance(ra, dict):
                     self._lift_actions_to_commands(ra)
                     ra_item.setData(ra, ROLE_DATA)
                card_item.appendRow(ra_item)

            keywords_data = card.get('keywords', {})
            if keywords_data and isinstance(keywords_data, dict):
                has_keywords_node = False
                for chk_i in range(card_item.rowCount()):
                    chk_child = card_item.child(chk_i)
                    if chk_child is not None and chk_child.data(ROLE_TYPE) == "KEYWORDS":
                        has_keywords_node = True
                        break

                if not has_keywords_node:
                    kw_item = QStandardItem(tr("Keywords"))
                    kw_item.setData("KEYWORDS", ROLE_TYPE)
                    kw_item.setData(keywords_data.copy(), ROLE_DATA)
                    card_item.appendRow(kw_item)

                if keywords_data.get('revolution_change'):
                    self.add_revolution_change_logic(card_item)
                if keywords_data.get('mekraid'):
                    self.add_mekraid_logic(card_item)
                if keywords_data.get('friend_burst'):
                    self.add_friend_burst_logic(card_item)

            spell_side_data = card.get('spell_side')
            if spell_side_data:
                if 'triggers' in spell_side_data:
                    spell_side_data['_meta_use_triggers'] = True

                spell_item = self._create_spell_side_item(spell_side_data)
                spell_triggers = spell_side_data.get('triggers', [])
                if not spell_triggers:
                    spell_triggers = spell_side_data.get('effects', [])

                for eff_idx, effect in enumerate(spell_triggers):
                    self._lift_actions_to_commands(effect)
                    eff_item = self._create_effect_item(effect)
                    self._load_effect_children(eff_item, effect)
                    spell_item.appendRow(eff_item)

                for mod_idx, modifier in enumerate(spell_side_data.get('static_abilities', [])):
                     mod_item = self._create_modifier_item(modifier)
                     spell_item.appendRow(mod_item)

                card_item.appendRow(spell_item)

            self.model.appendRow(card_item)
            try:
                updated = self._update_card_from_child(card_item)
                if updated:
                    self._sync_editor_warnings(card_item)
            except Exception:
                pass

    def _load_effect_children(self, eff_item, effect_data):
        for cmd_idx, command in enumerate(effect_data.get('commands', [])):
            cmd_item = self.create_command_item(command)
            eff_item.appendRow(cmd_item)

    def get_full_data(self):
        cards: list[JSON] = []
        root = self.model.invisibleRootItem()
        if root is None:
            return cards
        for i in range(root.rowCount()):
            card_item = root.child(i)
            if card_item is None:
                continue
            card_data = self.reconstruct_card_data(card_item)
            if card_data:
                warnings = self._normalize_card_for_engine(card_data)
                model_errors = self.validate_card_data(card_data)
                if model_errors:
                     warnings.extend([f"[Strict Model] {e}" for e in model_errors])

                if warnings:
                    card_data.setdefault('_editor_warnings', []).extend(warnings)
                cards.append(card_data)
        return cards

    def _normalize_card_for_engine(self, card: dict) -> list:
        warnings = []
        from dm_toolkit.consts import COMMAND_TYPES
        from dm_toolkit.gui.editor.validators_shared import (
            ModifierValidator, TriggerEffectValidator, FilterValidator, ConditionValidator
        )
        from dm_toolkit.gui.editor.data_migration import migrate_card_data

        migrated_count = migrate_card_data(card)
        if migrated_count > 0:
            warnings.append(f"自動マイグレーション: {migrated_count}個のフィールドを更新しました")

        def _ensure_uid(obj):
            if isinstance(obj, dict) and 'uid' not in obj:
                obj['uid'] = str(uuid.uuid4())

        def _normalize_command(cmd, path):
            w = []
            if not isinstance(cmd, dict):
                return None, [f"無効なコマンド: {path}（オブジェクトではありません）"]
            _ensure_uid(cmd)
            ctype = cmd.get('type')
            if not ctype or ctype not in COMMAND_TYPES:
                cmd['legacy_warning'] = True
                w.append(f"不明なコマンド種別: {path}: {ctype}")

            for key in ('if_true', 'if_false', 'options'):
                if key in cmd:
                    val = cmd.get(key)
                    if not isinstance(val, list):
                        w.append(f"フィールド '{key}' はリストである必要があります: {path}（自動修正しました）")
                        cmd[key] = []
                        continue
                    new_list = []
                    for idx, sub in enumerate(val):
                        if key == 'options':
                            if not isinstance(sub, list):
                                w.append(f"選択肢 {path}.{key}[{idx}] がリストではありません（スキップしました）")
                                continue
                            opt_cmds = []
                            for jdx, ssub in enumerate(sub):
                                normalized, subw = _normalize_command(ssub, f"{path}.{key}[{idx}][{jdx}]")
                                if normalized is not None:
                                    opt_cmds.append(normalized)
                                w.extend(subw)
                            new_list.append(opt_cmds)
                        else:
                            normalized, subw = _normalize_command(sub, f"{path}.{key}[{idx}]")
                            if normalized is not None:
                                new_list.append(normalized)
                            w.extend(subw)
                    cmd[key] = new_list
            return cmd, w

        effects_key = 'triggers' if 'triggers' in card else 'effects'
        effects = card.get(effects_key, [])
        for ei, eff in enumerate(effects):
            eff_path = f"card.{effects_key}[{ei}]"
            trigger_errors = TriggerEffectValidator.validate(eff)
            for err in trigger_errors:
                warnings.append(f"{eff_path}: {err}")
            
            cmds = eff.get('commands', [])
            if cmds and not isinstance(cmds, list):
                warnings.append(f"'commands' in {eff_path} is not a list; clearing.")
                eff['commands'] = []
                cmds = []
            new_cmds = []
            for ci, cmd in enumerate(cmds):
                normalized, w = _normalize_command(cmd, f"{eff_path}.commands[{ci}]")
                if normalized is not None:
                    new_cmds.append(normalized)
                warnings.extend(w)
            eff['commands'] = new_cmds

        static_abilities = card.get('static_abilities', [])
        for si, mod in enumerate(static_abilities):
            mod_path = f"card.static_abilities[{si}]"
            modifier_errors = ModifierValidator.validate(mod)
            for err in modifier_errors:
                warnings.append(f"{mod_path}: {err}")
            
            try:
                from dm_toolkit.consts import TargetScope
                scope_val = mod.get('scope')
                if scope_val:
                    scope_norm = TargetScope.normalize(scope_val)
                    if scope_norm and scope_norm != TargetScope.ALL:
                        filt = mod.get('filter') or {}
                        owner = filt.get('owner')
                        if not owner:
                            filt['owner'] = scope_norm
                            mod['filter'] = filt
                            warnings.append(f"{mod_path}: Applied scope '{scope_norm}' to filter.owner for engine")
            except Exception:
                pass

            if 'filter' in mod:
                filter_errors = FilterValidator.validate(mod.get('filter', {}))
                for err in filter_errors:
                    warnings.append(f"{mod_path}.filter: {err}")
            if 'condition' in mod:
                cond_errors = ConditionValidator.validate_static(mod.get('condition', {}))
                for err in cond_errors:
                    warnings.append(f"{mod_path}.condition: {err}")
        
        reaction_abilities = card.get('reaction_abilities', [])
        for ri, ra in enumerate(reaction_abilities):
            ra_path = f"card.reaction_abilities[{ri}]"
            if 'condition' in ra:
                cond_errors = ConditionValidator.validate_trigger(ra.get('condition', {}))
                for err in cond_errors:
                    warnings.append(f"{ra_path}.condition: {err}")

        spell_side = card.get('spell_side')
        if spell_side:
            spell_effects_key = 'triggers' if 'triggers' in spell_side else 'effects'
            spell_effects = spell_side.get(spell_effects_key, [])
            for ei, eff in enumerate(spell_effects):
                eff_path = f"card.spell_side.{spell_effects_key}[{ei}]"
                trigger_errors = TriggerEffectValidator.validate(eff)
                for err in trigger_errors:
                    warnings.append(f"{eff_path}: {err}")
                
                cmds = eff.get('commands', [])
                if cmds and not isinstance(cmds, list):
                    warnings.append(f"'commands' in {eff_path} is not a list; clearing.")
                    eff['commands'] = []
                    cmds = []
                new_cmds = []
                for ci, cmd in enumerate(cmds):
                    normalized, w = _normalize_command(cmd, f"{eff_path}.commands[{ci}]")
                    if normalized is not None:
                        new_cmds.append(normalized)
                    warnings.extend(w)
                eff['commands'] = new_cmds
            
            spell_static = spell_side.get('static_abilities', [])
            for si, mod in enumerate(spell_static):
                mod_path = f"card.spell_side.static_abilities[{si}]"
                modifier_errors = ModifierValidator.validate(mod)
                for err in modifier_errors:
                    warnings.append(f"{mod_path}: {err}")

        return warnings

    def reconstruct_card_data(self, card_item):
        card_data = card_item.data(ROLE_DATA)
        if not card_data:
            return None

        new_effects = []
        new_static = []
        new_reactions = []
        spell_side_dict = None
        keywords_dict = {}

        rev_change_filter = None
        has_rev_change_action = False
        friend_burst_filter = None
        has_friend_burst_action = False

        for j in range(card_item.rowCount()):
            child_item = card_item.child(j)
            item_type = child_item.data(ROLE_TYPE)

            if isinstance(item_type, str) and item_type.startswith("GROUP_"):
                for k in range(child_item.rowCount()):
                    grp_child = child_item.child(k)
                    if grp_child is None:
                        continue
                    if grp_child.data(ROLE_TYPE) == "EFFECT":
                        eff_data = self._reconstruct_effect(grp_child)
                        new_effects.append(eff_data)
                        for act in eff_data.get('commands', []) or []:
                            if isinstance(act, dict) and act.get('mutation_kind') == "REVOLUTION_CHANGE":
                                has_rev_change_action = True
                                rev_change_filter = act.get('target_filter') or act.get('filter')
                            if isinstance(act, dict) and act.get('type') == "REVOLUTION_CHANGE":
                                has_rev_change_action = True
                                rev_change_filter = act.get('filter')
                            if isinstance(act, dict) and act.get('type') == 'FRIEND_BURST':
                                has_friend_burst_action = True
                                friend_burst_filter = act.get('target_filter') or act.get('filter')
                continue

            if item_type == "KEYWORDS":
                kw_data = child_item.data(ROLE_DATA)
                if kw_data:
                    keywords_dict = kw_data.copy()

            elif item_type == "EFFECT":
                eff_data = self._reconstruct_effect(child_item)
                new_effects.append(eff_data)
                for act in eff_data.get('commands', []) or []:
                    if isinstance(act, dict) and act.get('mutation_kind') == "REVOLUTION_CHANGE":
                        has_rev_change_action = True
                        rev_change_filter = act.get('target_filter') or act.get('filter')
                    if isinstance(act, dict) and act.get('type') == "REVOLUTION_CHANGE":
                        has_rev_change_action = True
                        rev_change_filter = act.get('filter')
                    if isinstance(act, dict) and act.get('type') == 'FRIEND_BURST':
                        has_friend_burst_action = True
                        friend_burst_filter = act.get('target_filter') or act.get('filter')

            elif item_type == "MODIFIER":
                new_static.append(self._reconstruct_modifier(child_item))

            elif item_type == "REACTION_ABILITY":
                new_reactions.append(child_item.data(ROLE_DATA))

            elif item_type == "SPELL_SIDE":
                spell_side_data = child_item.data(ROLE_DATA)
                spell_side_effects = []
                spell_side_static = []

                for k in range(child_item.rowCount()):
                    sp_child = child_item.child(k)
                    sp_type = sp_child.data(ROLE_TYPE)

                    if sp_type == "EFFECT":
                        spell_side_effects.append(self._reconstruct_effect(sp_child))

                    elif sp_type == "MODIFIER":
                        spell_side_static.append(self._reconstruct_modifier(sp_child))

                if spell_side_data.get('_meta_use_triggers'):
                    spell_side_data['triggers'] = spell_side_effects
                    if 'effects' in spell_side_data:
                        del spell_side_data['effects']
                    del spell_side_data['_meta_use_triggers']
                else:
                    spell_side_data['effects'] = spell_side_effects
                    if 'triggers' in spell_side_data:
                        del spell_side_data['triggers']

                if spell_side_static:
                    spell_side_data['static_abilities'] = spell_side_static

                spell_side_dict = spell_side_data

        if card_data.get('_meta_use_triggers'):
            card_data['triggers'] = new_effects
            if 'effects' in card_data:
                del card_data['effects']
            del card_data['_meta_use_triggers']
        else:
            card_data['effects'] = new_effects
            if 'triggers' in card_data:
                del card_data['triggers']

        card_data['static_abilities'] = new_static
        card_data['reaction_abilities'] = new_reactions

        if spell_side_dict:
            card_data['spell_side'] = spell_side_dict
        else:
            if 'spell_side' in card_data:
                del card_data['spell_side']

        current_keywords = card_data.get('keywords', {})
        current_keywords.update(keywords_dict)
        card_data['keywords'] = current_keywords

        if has_rev_change_action and rev_change_filter:
            card_data['keywords']['revolution_change'] = True
            card_data['revolution_change_condition'] = rev_change_filter
        else:
            if 'revolution_change_condition' in card_data:
                del card_data['revolution_change_condition']

        if has_friend_burst_action and friend_burst_filter:
            card_data['keywords']['friend_burst'] = True
            card_data['friend_burst_condition'] = friend_burst_filter
        else:
            if 'friend_burst_condition' in card_data:
                del card_data['friend_burst_condition']

        return card_data

    def _reconstruct_effect(self, eff_item):
        eff_data = eff_item.data(ROLE_DATA)
        new_commands = []
        for k in range(eff_item.rowCount()):
            item = eff_item.child(k)
            item_type = item.data(ROLE_TYPE)
            if item_type == "ACTION":
                act_data = self._reconstruct_action(item)
                try:
                    objs = convert_action_to_objs(act_data)
                    for o in objs:
                        if hasattr(o, 'to_dict'):
                            new_commands.append(o.to_dict())
                        elif isinstance(o, dict):
                            new_commands.append(o)
                        else:
                            new_commands.append({
                                'type': 'NONE',
                                'legacy_warning': True,
                                'legacy_original_action': act_data
                            })
                except Exception:
                    new_commands.append({
                        'type': 'NONE',
                        'legacy_warning': True,
                        'legacy_original_action': act_data
                    })

            elif item_type == "COMMAND":
                cmd_data = self._reconstruct_command(item)
                new_commands.append(cmd_data)

        if new_commands:
            eff_data['commands'] = new_commands
        else:
            if 'commands' in eff_data:
                del eff_data['commands']
        if 'actions' in eff_data:
            del eff_data['actions']
        return eff_data

    def _reconstruct_action(self, act_item):
        # Deprecated
        act_data = act_item.data(ROLE_DATA)
        if act_item.rowCount() > 0:
            options = []
            for m in range(act_item.rowCount()):
                option_item = act_item.child(m)
                if option_item.data(ROLE_TYPE) == "OPTION":
                    option_actions = []
                    for n in range(option_item.rowCount()):
                        sub_act_item = option_item.child(n)
                        item_type = sub_act_item.data(ROLE_TYPE)
                        if item_type == "ACTION":
                            option_actions.append(self._reconstruct_action(sub_act_item))
                        elif item_type == "COMMAND":
                            option_actions.append(self._reconstruct_command(sub_act_item))
                    options.append(option_actions)
            act_data['options'] = options
        elif 'options' in act_data:
            del act_data['options']
        return act_data

    def _reconstruct_modifier(self, mod_item):
        return mod_item.data(ROLE_DATA)

    def _reconstruct_command(self, cmd_item):
        cmd_data = cmd_item.data(ROLE_DATA)
        if_true_list = []
        if_false_list = []
        options_list = []

        for i in range(cmd_item.rowCount()):
            child = cmd_item.child(i)
            role = child.data(ROLE_TYPE)

            if role == "CMD_BRANCH_TRUE":
                for j in range(child.rowCount()):
                    sub_item = child.child(j)
                    if sub_item.data(ROLE_TYPE) == "COMMAND":
                        if_true_list.append(self._reconstruct_command(sub_item))
            elif role == "CMD_BRANCH_FALSE":
                for j in range(child.rowCount()):
                    sub_item = child.child(j)
                    if sub_item.data(ROLE_TYPE) == "COMMAND":
                        if_false_list.append(self._reconstruct_command(sub_item))
            elif role == "OPTION":
                opt_cmds = []
                for j in range(child.rowCount()):
                    sub_item = child.child(j)
                    if sub_item.data(ROLE_TYPE) == "COMMAND":
                        opt_cmds.append(self._reconstruct_command(sub_item))
                options_list.append(opt_cmds)

        if if_true_list: cmd_data['if_true'] = if_true_list
        elif 'if_true' in cmd_data: del cmd_data['if_true']

        if if_false_list: cmd_data['if_false'] = if_false_list
        elif 'if_false' in cmd_data: del cmd_data['if_false']

        if options_list: cmd_data['options'] = options_list
        elif 'options' in cmd_data: del cmd_data['options']

        return cmd_data

    def add_new_card(self):
        new_id = self._generate_new_id()
        new_card = {
            "id": new_id, "name": "新規カード",
            "civilizations": ["FIRE"], "type": "CREATURE",
            "cost": 1, "power": 1000, "races": [], "effects": []
        }
        item = self._create_card_item(new_card)
        self.model.appendRow(item)
        return item

    def add_child_item(self, parent_index, item_type, data, label):
        if not parent_index.isValid(): return None
        parent_item = self.model.itemFromIndex(parent_index)
        if parent_item is None:
            return None
        target_item = parent_item

        if 'uid' not in data:
            data['uid'] = str(uuid.uuid4())

        if item_type == "ACTION":
            try:
                objs = convert_action_to_objs(data)
                created_item = None
                for o in objs:
                    if isinstance(o, WarningCommand):
                        continue
                    if hasattr(o, 'to_dict'):
                        cmd_dict = o.to_dict()
                    elif isinstance(o, dict):
                        cmd_dict = o
                    else:
                        continue
                    if 'uid' not in cmd_dict:
                        cmd_dict['uid'] = data.get('uid')
                    cmd_item = self.create_command_item(cmd_dict)
                    target_item.appendRow(cmd_item)
                    if not created_item: created_item = cmd_item
                if created_item:
                    return created_item
            except Exception:
                pass

        new_item = QStandardItem(label)
        new_item.setData(item_type, ROLE_TYPE)
        new_item.setData(data, ROLE_DATA)
        target_item.appendRow(new_item)
        return new_item

    def add_spell_side_item(self, card_item):
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child is None:
                continue
            if child.data(ROLE_TYPE) == "SPELL_SIDE":
                return child

        spell_data = {
            "name": "New Spell Side", "type": "SPELL",
            "cost": 1, "civilizations": [], "effects": []
        }
        item = self._create_spell_side_item(spell_data)
        card_item.appendRow(item)
        try:
            self._update_card_from_child(card_item)
        except Exception:
            pass
        return item

    def remove_spell_side_item(self, card_item):
        for i in reversed(range(card_item.rowCount())):
            child = card_item.child(i)
            if child.data(ROLE_TYPE) == "SPELL_SIDE":
                card_item.removeRow(i)
                try:
                    self._update_card_from_child(card_item)
                except Exception:
                    pass
                return True
        return False

    def add_revolution_change_logic(self, card_item):
        return self._add_logic_from_template("revolution_change", card_item, 'mutation_kind', 'REVOLUTION_CHANGE')

    def remove_revolution_change_logic(self, card_item):
        rows_to_remove = []
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(ROLE_TYPE) == "EFFECT":
                eff_data = child.data(ROLE_DATA)
                for cmd in eff_data.get('commands', []):
                    if cmd.get('mutation_kind') == 'REVOLUTION_CHANGE':
                        rows_to_remove.append(i)
                        break
        for i in reversed(rows_to_remove):
            card_item.removeRow(i)

    def add_mekraid_logic(self, card_item):
        return self._add_logic_from_template("mekraid", card_item, 'type', 'MEKRAID')

    def remove_mekraid_logic(self, card_item):
        rows_to_remove = []
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(ROLE_TYPE) == "EFFECT":
                eff_data = child.data(ROLE_DATA)
                for cmd in eff_data.get('commands', []):
                    if cmd.get('type') == 'MEKRAID':
                        rows_to_remove.append(i)
                        break
        for i in reversed(rows_to_remove):
            card_item.removeRow(i)

    def add_friend_burst_logic(self, card_item):
        return self._add_logic_from_template("friend_burst", card_item, 'type', 'FRIEND_BURST')

    def _add_logic_from_template(self, template_key, card_item, check_key, check_val):
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(ROLE_TYPE) == "EFFECT":
                eff_data = child.data(ROLE_DATA)
                for cmd in eff_data.get('commands', []):
                    if cmd.get(check_key) == check_val:
                        return child

        try:
            card_data = card_item.data(ROLE_DATA) or {}
        except Exception:
            card_data = {}

        context = {
            'civilizations': list(card_data.get('civilizations', ["FIRE"])),
            'races': list(card_data.get('races', ["Dragon"]))
        }

        eff_data, keywords, meta = self.template_manager.apply_template(template_key, context)
        if not eff_data:
            return None

        eff_item = self._create_effect_item(eff_data)
        self._load_effect_children(eff_item, eff_data)

        attached = False
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(ROLE_TYPE) == "GROUP_TRIGGER":
                child.appendRow(eff_item)
                attached = True
                break

        if not attached:
            card_item.appendRow(eff_item)

        try:
            self._update_card_from_child(eff_item)
            card_data = card_item.data(ROLE_DATA) or {}
            current_keywords = card_data.get('keywords', {})
            current_keywords.update(keywords)
            card_data['keywords'] = current_keywords

            mappings = meta.get('condition_mapping', {})
            for root_key, path in mappings.items():
                if path.startswith("commands[0]."):
                    sub_key = path.split(".")[1]
                    if eff_data.get('commands') and len(eff_data['commands']) > 0:
                        val = eff_data['commands'][0].get(sub_key)
                        if val:
                            card_data[root_key] = val

            card_item.setData(card_data, ROLE_DATA)
        except Exception:
            pass
        return eff_item

    def remove_friend_burst_logic(self, card_item):
        rows_to_remove = []
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(ROLE_TYPE) == "EFFECT":
                eff_data = child.data(ROLE_DATA)
                for cmd in eff_data.get('commands', []):
                    if cmd.get('type') == 'FRIEND_BURST':
                        rows_to_remove.append(i)
                        break
        for i in reversed(rows_to_remove):
            card_item.removeRow(i)

    def add_option_slots(self, action_item, count):
        current_options = 0
        for i in range(action_item.rowCount()):
             if action_item.child(i).data(ROLE_TYPE) == "OPTION":
                  current_options += 1
        for i in range(count):
            opt_num = current_options + i + 1
            opt_item = QStandardItem(f"{tr('Option')} {opt_num}")
            opt_item.setData("OPTION", ROLE_TYPE)
            uid = str(uuid.uuid4())
            opt_item.setData({'uid': uid}, ROLE_DATA)
            try:
                self._internalize_item(opt_item)
            except Exception:
                pass
            action_item.appendRow(opt_item)
        try:
            self._update_card_from_child(action_item)
        except Exception:
            pass

    def add_command_branches(self, cmd_item):
        has_true, has_false = False, False
        for i in range(cmd_item.rowCount()):
            role = cmd_item.child(i).data(ROLE_TYPE)
            if role == "CMD_BRANCH_TRUE": has_true = True
            if role == "CMD_BRANCH_FALSE": has_false = True

        if not has_true:
            true_item = QStandardItem(tr("If True"))
            true_item.setData("CMD_BRANCH_TRUE", ROLE_TYPE)
            t_uid = str(uuid.uuid4())
            true_item.setData({'uid': t_uid}, ROLE_DATA)
            try:
                self._internalize_item(true_item)
            except Exception:
                pass
            cmd_item.appendRow(true_item)

        if not has_false:
            false_item = QStandardItem(tr("If False"))
            false_item.setData("CMD_BRANCH_FALSE", ROLE_TYPE)
            f_uid = str(uuid.uuid4())
            false_item.setData({'uid': f_uid}, ROLE_DATA)
            try:
                self._internalize_item(false_item)
            except Exception:
                pass
            cmd_item.appendRow(false_item)

    def _create_card_item(self, card):
        if 'uid' not in card:
            card['uid'] = str(uuid.uuid4())
        item = QStandardItem(f"{card.get('id')} - {card.get('name', 'No Name')}")
        item.setData("CARD", ROLE_TYPE)
        item.setData(card, ROLE_DATA)

        kw_item = QStandardItem(tr("Keywords"))
        kw_item.setData("KEYWORDS", ROLE_TYPE)
        kw_item.setData(card.get('keywords', {}), ROLE_DATA)
        kw_item.setEditable(False)
        item.appendRow(kw_item)

        try:
            self._internalize_item(item)
        except Exception:
            pass
        return item

    def _create_spell_side_item(self, spell_data):
        self._ensure_uid(spell_data)
        item = QStandardItem(f"{tr('Spell Side')}: {spell_data.get('name', 'No Name')}")
        item.setData("SPELL_SIDE", ROLE_TYPE)
        item.setData(spell_data, ROLE_DATA)
        return item

    def _create_effect_item(self, effect):
        self._ensure_uid(effect)
        trig = effect.get('trigger', 'NONE')
        item = QStandardItem(f"{tr('Effect')}: {tr(trig)}")
        item.setData("EFFECT", ROLE_TYPE)
        item.setData(effect, ROLE_DATA)
        try:
            self._internalize_item(item)
        except Exception:
            pass
        return item

    def _create_modifier_item(self, modifier):
        self._ensure_uid(modifier)
        mtype = modifier.get('type', 'NONE')
        item = QStandardItem(f"{tr('Static')}: {tr(mtype)}")
        item.setData("MODIFIER", ROLE_TYPE)
        item.setData(modifier, ROLE_DATA)
        return item

    def _create_reaction_item(self, reaction):
        self._ensure_uid(reaction)
        rtype = reaction.get('type', 'NONE')
        item = QStandardItem(f"{tr('Reaction Ability')}: {rtype}")
        item.setData("REACTION_ABILITY", ROLE_TYPE)
        item.setData(reaction, ROLE_DATA)
        return item

    def format_command_label(self, command):
        cmd_type = command.get('type', 'NONE')
        if cmd_type == 'MUTATE' and command.get('mutation_kind') == 'REVOLUTION_CHANGE':
            label = f"{tr('Action')}: {tr('REVOLUTION_CHANGE')}"
        else:
            label = f"{tr('Action')}: {tr(cmd_type)}"
        if command.get('legacy_warning'):
             label += " [WARNING: Incomplete Conversion]"
        return label

    def _create_action_item(self, action):
        # Deprecated
        return self._create_action_item_deprecated(action)

    def _create_action_item_deprecated(self, action):
        if 'uid' not in action:
            action['uid'] = str(uuid.uuid4())
        try:
            objs = convert_action_to_objs(action)
            for o in objs:
                if isinstance(o, WarningCommand):
                    continue
                if hasattr(o, 'to_dict'):
                    cmd_dict = o.to_dict()
                elif isinstance(o, dict):
                    cmd_dict = o
                else:
                    continue
                if 'uid' not in cmd_dict:
                    cmd_dict['uid'] = action.get('uid')
                return self.create_command_item(cmd_dict)
        except Exception as e:
            pass
        return self.create_command_item({
            'type': 'NONE',
            'legacy_warning': True,
            'warning': 'Conversion failed',
            'legacy_original_action': action,
            'uid': action.get('uid')
        })

    def create_command_item(self, command):
        # Check if command is a model wrapper
        if hasattr(command, 'to_dict'):
            command = command.to_dict()

        if 'uid' not in command:
            command['uid'] = str(uuid.uuid4())

        label = self.format_command_label(command)
        if command.get('legacy_warning'):
             label = f"⚠️ {label}"

        item = QStandardItem(label)
        item.setData("COMMAND", ROLE_TYPE)
        item.setData(command, ROLE_DATA)

        if command.get('legacy_warning'):
             item.setToolTip(
                 tr("Legacy Action: {orig}\nPlease replace with modern Commands.").format(
                     orig=command.get('legacy_original_type', 'Unknown')
                 )
             )

        if 'if_true' in command and command['if_true']:
            true_item = QStandardItem(tr("If True"))
            true_item.setData("CMD_BRANCH_TRUE", ROLE_TYPE)
            true_item.setData({'uid': str(uuid.uuid4())}, ROLE_DATA)
            item.appendRow(true_item)
            for child in command['if_true']:
                true_item.appendRow(self.create_command_item(child))

        if 'if_false' in command and command['if_false']:
            false_item = QStandardItem(tr("If False"))
            false_item.setData("CMD_BRANCH_FALSE", ROLE_TYPE)
            false_item.setData({'uid': str(uuid.uuid4())}, ROLE_DATA)
            item.appendRow(false_item)
            for child in command['if_false']:
                false_item.appendRow(self.create_command_item(child))

        if 'options' in command and command['options']:
            for i, opt_cmds in enumerate(command['options']):
                opt_item = QStandardItem(f"{tr('Option')} {i+1}")
                opt_item.setData("OPTION", ROLE_TYPE)
                opt_item.setData({'uid': str(uuid.uuid4())}, ROLE_DATA)
                item.appendRow(opt_item)
                for sub_cmd in opt_cmds:
                    opt_item.appendRow(self.create_command_item(sub_cmd))
        try:
            self._internalize_item(item)
        except Exception:
            pass
        return item

    def _generate_new_id(self):
        max_id = 0
        root = self.model.invisibleRootItem()
        if root is None:
            return 1
        for i in range(root.rowCount()):
            card_item = root.child(i)
            if card_item is None:
                continue
            card_data = card_item.data(ROLE_DATA)
            if card_data and 'id' in card_data:
                try:
                    cid = int(card_data['id'])
                    if cid > max_id: max_id = cid
                except ValueError:
                    pass
        return max_id + 1

    def _find_card_root(self, item):
        cur_item = item
        while cur_item is not None:
            role = cur_item.data(ROLE_TYPE)
            if role == 'CARD':
                return cur_item
            parent = cur_item.parent()
            if parent is None:
                return None
            cur_item = parent

    def _update_card_from_child(self, child_item):
        card_item = self._find_card_root(child_item)
        if card_item is None:
            return None
        try:
            updated = self.reconstruct_card_data(card_item)
            if updated:
                card_item.setData(updated, ROLE_DATA)
                return updated
        except Exception:
            pass
        return None

    def _find_child_by_role(self, parent_item, role_string):
        for i in range(parent_item.rowCount()):
            child = parent_item.child(i)
            if child.data(ROLE_TYPE) == role_string:
                return child
        return None

    def _sync_editor_warnings(self, card_item):
        card_data = card_item.data(ROLE_DATA)
        if not isinstance(card_data, dict):
            return
        warnings_list = card_data.get('_editor_warnings', [])

        warn_node = self._find_child_by_role(card_item, 'EDITOR_WARNINGS')
        if not warnings_list:
            if warn_node is not None:
                for i in range(card_item.rowCount()):
                    if card_item.child(i) is warn_node:
                        card_item.removeRow(i)
                        break
            return

        if warn_node is None:
            warn_node = QStandardItem(tr("Warnings"))
            warn_node.setData('EDITOR_WARNINGS', ROLE_TYPE)
            warn_node.setData({'uid': str(uuid.uuid4())}, ROLE_DATA)
            warn_node.setEditable(False)
            card_item.insertRow(0, warn_node)
        else:
            for i in reversed(range(warn_node.rowCount())):
                warn_node.removeRow(i)

        for w in warnings_list:
            w_item = QStandardItem(str(w))
            w_item.setData('EDITOR_WARNING', ROLE_TYPE)
            w_item.setData({'uid': str(uuid.uuid4()), 'text': w}, ROLE_DATA)
            w_item.setEditable(False)
            warn_node.appendRow(w_item)

    def get_item_path(self, item):
        path = []
        curr = item
        root = self.model.invisibleRootItem()
        while curr and curr != root:
            data = self.get_item_data(curr)
            if data and 'uid' in data:
                path.append(f"uid_{data['uid']}")
            else:
                path.append(f"row_{curr.row()}")
            curr = curr.parent()
        return ":".join(reversed(path))

    def convert_action_tree_to_command(self, action_item):
        from dm_toolkit.gui.editor.action_converter import ActionConverter
        act_data = self.get_item_data(action_item)
        cmd_data = ActionConverter.convert(act_data)
        options_list = []
        if action_item.rowCount() > 0:
            for i in range(action_item.rowCount()):
                child = action_item.child(i)
                if child is None: continue
                child_type = self.get_item_type(child)
                if child_type == "OPTION":
                    opt_cmds = []
                    for k in range(child.rowCount()):
                        sub_item = child.child(k)
                        if sub_item is None: continue
                        sub_type = self.get_item_type(sub_item)
                        if sub_type == "ACTION":
                            opt_cmds.append(self.convert_action_tree_to_command(sub_item))
                        elif sub_type == "COMMAND":
                            opt_cmds.append(self.get_item_data(sub_item))
                    options_list.append(opt_cmds)
        if options_list:
            cmd_data['options'] = options_list
        return cmd_data

    def replace_action_with_command(self, index, cmd_data):
        if not index.isValid(): return None
        parent_item = self.model.itemFromIndex(index.parent())
        old_item = self.model.itemFromIndex(index)
        if old_item is None: return None
        row = index.row()

        preserved_options_data = []
        if old_item.rowCount() > 0:
            for i in range(old_item.rowCount()):
                child = old_item.child(i)
                if child is None: continue
                if self.get_item_type(child) == "OPTION":
                    opt_actions_data = []
                    for k in range(child.rowCount()):
                        act_child = child.child(k)
                        if act_child is None: continue
                        act_type = self.get_item_type(act_child)
                        if act_type == "ACTION":
                            c_cmd = self.convert_action_tree_to_command(act_child)
                            opt_actions_data.append(c_cmd)
                        elif act_type == "COMMAND":
                            opt_actions_data.append(self.get_item_data(act_child))
                    preserved_options_data.append(opt_actions_data)

        if preserved_options_data:
            cmd_data['options'] = preserved_options_data

        if parent_item is None:
            if not index.parent().isValid():
                parent_item = self.model.invisibleRootItem()
        if parent_item is None: return None

        parent_item.removeRow(row)
        cmd_item = self.create_command_item(cmd_data)
        parent_item.insertRow(row, cmd_item)
        return cmd_item

    def collect_conversion_preview(self, item):
        previews = []
        def _recurse(cur_item):
            for i in range(cur_item.rowCount()):
                child = cur_item.child(i)
                if child is None: continue
                child_type = self.get_item_type(child)
                if child_type == 'ACTION':
                    cmd_data = self.convert_action_tree_to_command(child)
                    warn = bool(cmd_data.get('legacy_warning', False))
                    previews.append({
                        'path': self.get_item_path(child),
                        'label': child.text(),
                        'warning': warn,
                        'cmd_data': cmd_data
                    })
                    for j in range(child.rowCount()):
                        opt = child.child(j)
                        if opt and self.get_item_type(opt) == 'OPTION':
                            _recurse(opt)
                else:
                    _recurse(child)
        _recurse(item)
        return previews

    def scan_warnings_in_cmd(self, cmd_data):
        w = 0
        if cmd_data.get('legacy_warning', False):
            w += 1
        if 'options' in cmd_data:
            for opt_list in cmd_data['options']:
                for sub_cmd in opt_list:
                    w += self.scan_warnings_in_cmd(sub_cmd)
        return w

    def batch_convert_actions_recursive(self, item):
        converted_count = 0
        warning_count = 0
        for i in reversed(range(item.rowCount())):
            child = item.child(i)
            if child is None: continue
            child_type = self.get_item_type(child)

            if child_type == "ACTION":
                cmd_data = self.convert_action_tree_to_command(child)
                w = self.scan_warnings_in_cmd(cmd_data)
                warning_count += w
                self.replace_action_with_command(child.index(), cmd_data)
                converted_count += 1
            else:
                c, w = self.batch_convert_actions_recursive(child)
                converted_count += c
                warning_count += w
        return converted_count, warning_count
