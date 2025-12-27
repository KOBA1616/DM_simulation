# -*- coding: utf-8 -*-
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr
import uuid
import json
import os
import copy
from dm_toolkit.types import JSON
from dm_toolkit.gui.editor import normalize
from dm_toolkit.gui.editor.action_converter import ActionConverter, convert_action_to_objs
from dm_toolkit.gui.editor.command_model import CommandDef, WarningCommand

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Handles loading, saving (reconstruction), and item creation (ID generation).
    """

    def __init__(self, model: QStandardItemModel):
        self.model = model
        self.templates: JSON = {"commands": [], "actions": []}
        self.load_templates()
        # cache can hold internal representations keyed by uid for editor-only use
        self._internal_cache = {}

    def load_templates(self):
        # Resolve path to data/editor_templates.json
        filepath = None

        # 1. Check Environment Variable
        env_path = os.environ.get('DM_EDITOR_TEMPLATES_PATH')
        if env_path and os.path.exists(env_path):
            filepath = env_path
        else:
            # 2. Search relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(current_dir, '..', '..', '..', 'data', 'editor_templates.json'),
                os.path.join(current_dir, '..', '..', 'data', 'editor_templates.json'),
                # Add explicit data path relative to CWD for robustness in various envs
                os.path.join(os.getcwd(), 'data', 'editor_templates.json')
            ]

            for candidate in candidates:
                if os.path.exists(candidate):
                    filepath = candidate
                    break

        if filepath and os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.templates = json.load(f)
            except Exception as e:
                print(f"Error loading templates: {e}")
                self.templates = {"commands": [], "actions": []}
        else:
            print("Warning: editor_templates.json not found.")

    def _ensure_uid(self, obj: dict):
        """Ensure a UID exists on the given dict object."""
        if obj is None or not isinstance(obj, dict):
            return
        if 'uid' not in obj:
            obj['uid'] = str(uuid.uuid4())

    def _internalize_item(self, item):
        """Create and cache an internal representation for the given tree item.
        This uses the normalize.to_internal heuristic and stores by uid.
        """
        try:
            data = item.data(Qt.ItemDataRole.UserRole + 2)
        except Exception:
            data = None

        # produce a canonical internal representation and cache it
        internal = normalize.canonicalize(data)
        uid = None
        if isinstance(data, dict):
            uid = data.get('uid')
        if not uid:
            # fallback: try to generate a temporary uid for caching
            uid = str(uuid.uuid4())

        self._internal_cache[uid] = internal
        return internal

    def get_internal_by_uid(self, uid: str):
        return self._internal_cache.get(uid)

    def _migrate_legacy_actions_in_effects(self, effects: list):
        """Helper to process a list of effect dicts and migrate 'actions' to 'commands'."""
        for eff in effects:
            try:
                legacy_actions = eff.get('actions', [])
                if legacy_actions:
                    converted_cmds = []
                    for act in list(legacy_actions):
                        try:
                            objs = convert_action_to_objs(act)
                            for o in objs:
                                # Convert CommandDef/WarningCommand to dict for storage
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
                    # Merge into any pre-existing commands list
                    eff['commands'] = eff.get('commands', []) + converted_cmds

                # Ensure 'commands' key exists on the effect (empty list if none)
                if 'commands' not in eff:
                    eff['commands'] = []

                # Remove legacy 'actions' to enforce Commands-only policy in-editor
                if 'actions' in eff:
                    try:
                        del eff['actions']
                    except Exception:
                        pass
            except Exception:
                # Non-fatal: if conversion fails, continue loading but preserve original actions
                pass

    def load_data(self, cards_data):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Logic Tree"])

        for card_idx, card in enumerate(cards_data):
            # 0. Check for 'triggers' key presence (Meta-info for preservation)
            if 'triggers' in card:
                card['_meta_use_triggers'] = True

            card_item = self._create_card_item(card)

            # 1. Add Creature Effects (Triggers)
            triggers = card.get('triggers', [])
            if not triggers:
                triggers = card.get('effects', [])

            # Apply Load-Lift Migration for main effects
            self._migrate_legacy_actions_in_effects(triggers)

            for eff_idx, effect in enumerate(triggers):
                eff_item = self._create_effect_item(effect)
                self._load_effect_children(eff_item, effect)
                card_item.appendRow(eff_item)

            # 1.5 Add Static Abilities
            for mod_idx, modifier in enumerate(card.get('static_abilities', [])):
                 mod_item = self._create_modifier_item(modifier)
                 card_item.appendRow(mod_item)

        # end load_data

            # 2. Add Reaction Abilities
            for ra_idx, ra in enumerate(card.get('reaction_abilities', [])):
                ra_item = self._create_reaction_item(ra)
                card_item.appendRow(ra_item)

            # 3. Add Spell Side if exists
            spell_side_data = card.get('spell_side')
            if spell_side_data:
                # Check for 'triggers' in spell side
                if 'triggers' in spell_side_data:
                    spell_side_data['_meta_use_triggers'] = True

                spell_item = self._create_spell_side_item(spell_side_data)

                # Add Spell Effects
                spell_triggers = spell_side_data.get('triggers', [])
                if not spell_triggers:
                    spell_triggers = spell_side_data.get('effects', [])

                # Apply Load-Lift Migration for spell side effects
                self._migrate_legacy_actions_in_effects(spell_triggers)

                for eff_idx, effect in enumerate(spell_triggers):
                    eff_item = self._create_effect_item(effect)
                    self._load_effect_children(eff_item, effect)
                    spell_item.appendRow(eff_item)

                # Add Spell Static (if any support in future)
                for mod_idx, modifier in enumerate(spell_side_data.get('static_abilities', [])):
                     mod_item = self._create_modifier_item(modifier)
                     spell_item.appendRow(mod_item)

                card_item.appendRow(spell_item)

            self.model.appendRow(card_item)
            # After building the card node, normalize and attach editor warnings node if any
            try:
                updated = self._update_card_from_child(card_item)
                if updated:
                    self._sync_editor_warnings(card_item)
            except Exception:
                pass

    def _load_effect_children(self, eff_item, effect_data):
        # Load Legacy Actions (if any remained or were re-added)
        for act_idx, action in enumerate(effect_data.get('actions', [])):
            act_item = self._create_action_item(action)
            eff_item.appendRow(act_item)

        # Load Commands
        for cmd_idx, command in enumerate(effect_data.get('commands', [])):
            cmd_item = self.create_command_item(command)
            eff_item.appendRow(cmd_item)

    def get_full_data(self):
        """Reconstructs the full JSON list from the tree model."""
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
                # Normalize and validate card for engine compatibility before returning
                warnings = self._normalize_card_for_engine(card_data)
                if warnings:
                    # Attach editor warnings so callers/UI can display them
                    card_data.setdefault('_editor_warnings', []).extend(warnings)
                cards.append(card_data)
        return cards

    def _normalize_card_for_engine(self, card: dict) -> list:
        """Normalize card JSON to be engine-friendly and return list of warnings.

        Fixes performed:
        - Ensure UIDs on commands and nested entries
        - Ensure command keys like 'options', 'if_true', 'if_false' are lists of dicts
        - Mark commands with missing/unknown 'type' as legacy_warning
        """
        warnings = []
        from dm_toolkit.consts import COMMAND_TYPES

        def _ensure_uid(obj):
            if isinstance(obj, dict) and 'uid' not in obj:
                obj['uid'] = str(uuid.uuid4())

        def _normalize_command(cmd, path):
            w = []
            if not isinstance(cmd, dict):
                return None, [f"Invalid command at {path}: not an object"]
            _ensure_uid(cmd)
            ctype = cmd.get('type')
            if not ctype or ctype not in COMMAND_TYPES:
                cmd['legacy_warning'] = True
                w.append(f"Unknown command type at {path}: {ctype}")

            # Normalize branches and options
            for key in ('if_true', 'if_false', 'options'):
                if key in cmd:
                    val = cmd.get(key)
                    if not isinstance(val, list):
                        w.append(f"Field '{key}' at {path} should be a list; fixing.")
                        cmd[key] = []
                        continue
                    new_list = []
                    for idx, sub in enumerate(val):
                        if key == 'options':
                            # options is a list of lists of commands
                            if not isinstance(sub, list):
                                w.append(f"Option entry at {path}.{key}[{idx}] not a list; skipping.")
                                continue
                            opt_cmds = []
                            for jdx, ssub in enumerate(sub):
                                normalized, subw = _normalize_command(ssub, f"{path}.{key}[{idx}][{jdx}]")
                                if normalized is not None:
                                    opt_cmds.append(normalized)
                                w.extend(subw)
                            new_list.append(opt_cmds)
                        else:
                            # branches: list of commands
                            normalized, subw = _normalize_command(sub, f"{path}.{key}[{idx}]")
                            if normalized is not None:
                                new_list.append(normalized)
                            w.extend(subw)
                    cmd[key] = new_list
            return cmd, w

        # Process effects/triggers
        effects_key = 'triggers' if 'triggers' in card else 'effects'
        effects = card.get(effects_key, [])
        for ei, eff in enumerate(effects):
            eff_path = f"card.effects[{ei}]"
            # Ensure lists
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

        return warnings

    def reconstruct_card_data(self, card_item):
        """Reconstructs a single card's data from its tree item."""
        card_data = card_item.data(Qt.ItemDataRole.UserRole + 2)
        if not card_data:
            return None

        new_effects = []
        new_static = []
        new_reactions = []
        spell_side_dict = None
        keywords_dict = {}

        # Revolution Change extraction
        rev_change_filter = None
        has_rev_change_action = False

        # Iterate children of CARD node (Flattened structure)
        for j in range(card_item.rowCount()):
            child_item = card_item.child(j)
            item_type = child_item.data(Qt.ItemDataRole.UserRole + 1)

            # Handle group containers (e.g., GROUP_TRIGGER holding EFFECT nodes)
            if isinstance(item_type, str) and item_type.startswith("GROUP_"):
                for k in range(child_item.rowCount()):
                    grp_child = child_item.child(k)
                    if grp_child is None:
                        continue
                    if grp_child.data(Qt.ItemDataRole.UserRole + 1) == "EFFECT":
                        eff_data = self._reconstruct_effect(grp_child)
                        new_effects.append(eff_data)
                        # Detect revolution change in either legacy actions or new commands
                        for act in (eff_data.get('actions', []) or []) + (eff_data.get('commands', []) or []):
                            if isinstance(act, dict) and act.get('type') == "REVOLUTION_CHANGE":
                                has_rev_change_action = True
                                rev_change_filter = act.get('filter')
                continue

            if item_type == "KEYWORDS":
                kw_data = child_item.data(Qt.ItemDataRole.UserRole + 2)
                if kw_data:
                    keywords_dict = kw_data.copy()

            elif item_type == "EFFECT":
                eff_data = self._reconstruct_effect(child_item)
                new_effects.append(eff_data)
                for act in (eff_data.get('actions', []) or []) + (eff_data.get('commands', []) or []):
                    if isinstance(act, dict) and act.get('type') == "REVOLUTION_CHANGE":
                        has_rev_change_action = True
                        rev_change_filter = act.get('filter')

            elif item_type == "MODIFIER":
                new_static.append(self._reconstruct_modifier(child_item))

            elif item_type == "REACTION_ABILITY":
                new_reactions.append(child_item.data(Qt.ItemDataRole.UserRole + 2))

            elif item_type == "SPELL_SIDE":
                # Reconstruct Spell Side
                spell_side_data = child_item.data(Qt.ItemDataRole.UserRole + 2)
                spell_side_effects = []
                spell_side_static = []

                # Iterate Spell Side Children (Flattened)
                for k in range(child_item.rowCount()):
                    sp_child = child_item.child(k)
                    sp_type = sp_child.data(Qt.ItemDataRole.UserRole + 1)

                    if sp_type == "EFFECT":
                        spell_side_effects.append(self._reconstruct_effect(sp_child))

                    elif sp_type == "MODIFIER":
                        spell_side_static.append(self._reconstruct_modifier(sp_child))

                # Handle Triggers vs Effects based on meta-info
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

        # Handle Triggers vs Effects based on meta-info
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

        # Merge keywords
        current_keywords = card_data.get('keywords', {})
        current_keywords.update(keywords_dict)
        card_data['keywords'] = current_keywords

        # Auto-set Revolution Change Keyword and Condition
        if has_rev_change_action and rev_change_filter:
            card_data['keywords']['revolution_change'] = True
            card_data['revolution_change_condition'] = rev_change_filter
        else:
            if 'revolution_change_condition' in card_data:
                del card_data['revolution_change_condition']

        return card_data

    def _reconstruct_effect(self, eff_item):
        eff_data = eff_item.data(Qt.ItemDataRole.UserRole + 2)

        # New policy: when reconstructing, prefer emitting `commands` only.
        # Convert any legacy ACTION nodes into command dicts (using ActionConverter).
        new_commands = []
        legacy_actions = []

        for k in range(eff_item.rowCount()):
            item = eff_item.child(k)
            item_type = item.data(Qt.ItemDataRole.UserRole + 1)

            if item_type == "ACTION":
                act_data = self._reconstruct_action(item)
                # preserve legacy action dict for backward compatibility
                legacy_actions.append(act_data)
                # Attempt conversion; always produce a command-like dict even on failure
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

        # Always output commands (if any). Do not emit legacy 'actions' in reconstructed JSON.
        if new_commands:
            eff_data['commands'] = new_commands
        else:
            if 'commands' in eff_data:
                del eff_data['commands']

        # Preserve legacy 'actions' list when original ACTION nodes existed
        if legacy_actions:
            eff_data['actions'] = legacy_actions
        else:
            if 'actions' in eff_data:
                del eff_data['actions']

        return eff_data

    def _reconstruct_action(self, act_item):
        act_data = act_item.data(Qt.ItemDataRole.UserRole + 2)
        if act_item.rowCount() > 0:
            options = []
            for m in range(act_item.rowCount()):
                option_item = act_item.child(m)
                if option_item.data(Qt.ItemDataRole.UserRole + 1) == "OPTION":
                    option_actions = []
                    for n in range(option_item.rowCount()):
                        sub_act_item = option_item.child(n)
                        if sub_act_item.data(Qt.ItemDataRole.UserRole + 1) == "ACTION":
                            option_actions.append(self._reconstruct_action(sub_act_item))
                    options.append(option_actions)
            act_data['options'] = options
        elif 'options' in act_data:
            del act_data['options']

        return act_data

    def _reconstruct_modifier(self, mod_item):
        return mod_item.data(Qt.ItemDataRole.UserRole + 2)

    def _reconstruct_command(self, cmd_item):
        cmd_data = cmd_item.data(Qt.ItemDataRole.UserRole + 2)
        if_true_list = []
        if_false_list = []
        options_list = []

        for i in range(cmd_item.rowCount()):
            child = cmd_item.child(i)
            role = child.data(Qt.ItemDataRole.UserRole + 1)

            if role == "CMD_BRANCH_TRUE":
                for j in range(child.rowCount()):
                    sub_item = child.child(j)
                    if sub_item.data(Qt.ItemDataRole.UserRole + 1) == "COMMAND":
                        if_true_list.append(self._reconstruct_command(sub_item))
            elif role == "CMD_BRANCH_FALSE":
                for j in range(child.rowCount()):
                    sub_item = child.child(j)
                    if sub_item.data(Qt.ItemDataRole.UserRole + 1) == "COMMAND":
                        if_false_list.append(self._reconstruct_command(sub_item))
            elif role == "OPTION":
                # Handle CHOICE options (list of commands)
                opt_cmds = []
                for j in range(child.rowCount()):
                    sub_item = child.child(j)
                    if sub_item.data(Qt.ItemDataRole.UserRole + 1) == "COMMAND":
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
            "id": new_id, "name": "New Card",
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
        parent_role = parent_item.data(Qt.ItemDataRole.UserRole + 1)

        target_item = parent_item

        # Create Item
        if 'uid' not in data:
            data['uid'] = str(uuid.uuid4())

        # If we're adding a legacy ACTION, attempt to convert it to COMMAND at creation time
        if item_type == "ACTION":
            try:
                objs = convert_action_to_objs(data)
                # If any of the converted objects is a non-warning CommandDef, create COMMAND nodes
                created = False
                for o in objs:
                    if isinstance(o, WarningCommand):
                        continue
                    # create command item from object dict
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
                    created = True
                if created:
                    return cmd_item
            except Exception:
                # Fall back to legacy action creation below
                pass

        new_item = QStandardItem(label)
        new_item.setData(item_type, Qt.ItemDataRole.UserRole + 1)
        new_item.setData(data, Qt.ItemDataRole.UserRole + 2)

        target_item.appendRow(new_item)
        return new_item

    def add_spell_side_item(self, card_item):
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child is None:
                continue
            if child.data(Qt.ItemDataRole.UserRole + 1) == "SPELL_SIDE":
                return child

        spell_data = {
            "name": "New Spell Side", "type": "SPELL",
            "cost": 1, "civilizations": [], "effects": []
        }
        item = self._create_spell_side_item(spell_data)
        card_item.appendRow(item)
        # Update card data stored on the CARD node
        try:
            self._update_card_from_child(card_item)
        except Exception:
            pass
        return item

    def remove_spell_side_item(self, card_item):
        for i in reversed(range(card_item.rowCount())):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "SPELL_SIDE":
                card_item.removeRow(i)
                try:
                    self._update_card_from_child(card_item)
                except Exception:
                    pass
                return True
        return False

    def add_revolution_change_logic(self, card_item):
        # Must add to card_item directly, searching for existing one
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "EFFECT":
                eff_data = child.data(Qt.ItemDataRole.UserRole + 2)
                for act in eff_data.get('actions', []):
                    if act.get('type') == 'REVOLUTION_CHANGE':
                        return child

        eff_data = {
            "trigger": "ON_ATTACK_FROM_HAND",
            "condition": {"type": "NONE"},
            "actions": []
        }
        eff_item = self._create_effect_item(eff_data)

        act_data = {
            "type": "REVOLUTION_CHANGE",
            "filter": {"civilizations": ["FIRE"], "races": ["Dragon"], "min_cost": 5}
        }
        act_item = self._create_action_item(act_data)
        eff_item.appendRow(act_item)

        # Prefer attaching to an existing GROUP_TRIGGER node if present
        attached = False
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "GROUP_TRIGGER":
                child.appendRow(eff_item)
                attached = True
                break

        if not attached:
            card_item.appendRow(eff_item)

        # Update reconstructed card data cache on change
        try:
            self._update_card_from_child(eff_item)
        except Exception:
            pass

        # Ensure revolution_change keyword is reflected immediately in the CARD data
        try:
            card_data = card_item.data(Qt.ItemDataRole.UserRole + 2) or {}
            if 'keywords' not in card_data or not isinstance(card_data['keywords'], dict):
                card_data['keywords'] = card_data.get('keywords', {}) or {}
            card_data['keywords']['revolution_change'] = True
            card_data['revolution_change_condition'] = act_data.get('filter')
            card_item.setData(card_data, Qt.ItemDataRole.UserRole + 2)
        except Exception:
            pass

        return eff_item

    def remove_revolution_change_logic(self, card_item):
        rows_to_remove = []
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "EFFECT":
                eff_data = child.data(Qt.ItemDataRole.UserRole + 2)
                for act in eff_data.get('actions', []):
                    if act.get('type') == 'REVOLUTION_CHANGE':
                        rows_to_remove.append(i)
                        break

        for i in reversed(rows_to_remove):
            card_item.removeRow(i)

    def add_option_slots(self, action_item, count):
        current_options = 0
        for i in range(action_item.rowCount()):
             if action_item.child(i).data(Qt.ItemDataRole.UserRole + 1) == "OPTION":
                  current_options += 1
        for i in range(count):
            opt_num = current_options + i + 1
            opt_item = QStandardItem(f"{tr('Option')} {opt_num}")
            opt_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
            uid = str(uuid.uuid4())
            opt_item.setData({'uid': uid}, Qt.ItemDataRole.UserRole + 2)
            # register internal representation for this newly created option
            try:
                self._internalize_item(opt_item)
            except Exception:
                pass
            action_item.appendRow(opt_item)
        # Update parent card data
        try:
            self._update_card_from_child(action_item)
        except Exception:
            pass

    def add_command_branches(self, cmd_item):
        has_true, has_false = False, False
        for i in range(cmd_item.rowCount()):
            role = cmd_item.child(i).data(Qt.ItemDataRole.UserRole + 1)
            if role == "CMD_BRANCH_TRUE": has_true = True
            if role == "CMD_BRANCH_FALSE": has_false = True

        if not has_true:
            true_item = QStandardItem(tr("If True"))
            true_item.setData("CMD_BRANCH_TRUE", Qt.ItemDataRole.UserRole + 1)
            t_uid = str(uuid.uuid4())
            true_item.setData({'uid': t_uid}, Qt.ItemDataRole.UserRole + 2)
            try:
                self._internalize_item(true_item)
            except Exception:
                pass
            cmd_item.appendRow(true_item)

        if not has_false:
            false_item = QStandardItem(tr("If False"))
            false_item.setData("CMD_BRANCH_FALSE", Qt.ItemDataRole.UserRole + 1)
            f_uid = str(uuid.uuid4())
            false_item.setData({'uid': f_uid}, Qt.ItemDataRole.UserRole + 2)
            try:
                self._internalize_item(false_item)
            except Exception:
                pass
            cmd_item.appendRow(false_item)
        # Update parent card data
        try:
            self._update_card_from_child(cmd_item)
        except Exception:
            pass

    def _create_card_item(self, card):
        if 'uid' not in card:
            card['uid'] = str(uuid.uuid4())
        item = QStandardItem(f"{card.get('id')} - {card.get('name', 'No Name')}")
        item.setData("CARD", Qt.ItemDataRole.UserRole + 1)
        item.setData(card, Qt.ItemDataRole.UserRole + 2)

        # Create Node Type 1: Keywords
        kw_item = QStandardItem(tr("Keywords"))
        kw_item.setData("KEYWORDS", Qt.ItemDataRole.UserRole + 1)
        # We pass the keywords dictionary explicitly as data for this item
        kw_item.setData(card.get('keywords', {}), Qt.ItemDataRole.UserRole + 2)
        kw_item.setEditable(False)
        item.appendRow(kw_item)

        # Cache internal representation
        try:
            self._internalize_item(item)
        except Exception:
            pass

        return item

    def _create_spell_side_item(self, spell_data):
        self._ensure_uid(spell_data)
        item = QStandardItem(f"{tr('Spell Side')}: {spell_data.get('name', 'No Name')}")
        item.setData("SPELL_SIDE", Qt.ItemDataRole.UserRole + 1)
        item.setData(spell_data, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_effect_item(self, effect):
        self._ensure_uid(effect)
        trig = effect.get('trigger', 'NONE')
        item = QStandardItem(f"{tr('Effect')}: {tr(trig)}")
        item.setData("EFFECT", Qt.ItemDataRole.UserRole + 1)
        item.setData(effect, Qt.ItemDataRole.UserRole + 2)
        try:
            self._internalize_item(item)
        except Exception:
            pass
        return item

    def _create_modifier_item(self, modifier):
        self._ensure_uid(modifier)
        mtype = modifier.get('type', 'NONE')
        item = QStandardItem(f"{tr('Static')}: {tr(mtype)}")
        item.setData("MODIFIER", Qt.ItemDataRole.UserRole + 1)
        item.setData(modifier, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_reaction_item(self, reaction):
        self._ensure_uid(reaction)
        rtype = reaction.get('type', 'NONE')
        item = QStandardItem(f"{tr('Reaction Ability')}: {rtype}")
        item.setData("REACTION_ABILITY", Qt.ItemDataRole.UserRole + 1)
        item.setData(reaction, Qt.ItemDataRole.UserRole + 2)
        return item


    def format_command_label(self, command):
        """Generates a human-readable label for a command."""
        cmd_type = command.get('type', 'NONE')
        # Use 'Action' instead of 'Command' for UI consistency
        label = f"{tr('Action')}: {tr(cmd_type)}"
        if command.get('legacy_warning'):
             label += " [WARNING: Incomplete Conversion]"
        return label

    def _create_action_item(self, action):
        # Ensure UID
        if 'uid' not in action:
            action['uid'] = str(uuid.uuid4())

        # Try to convert legacy Action -> Command via object adapter and prefer command representation
        try:
            objs = convert_action_to_objs(action)
            created = False
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
                # Return the first created command item (we could append multiple if needed)
                return self.create_command_item(cmd_dict)
        except Exception:
            # Conversion failed; fall back to creating an ACTION item
            pass

        # Fallback: keep as legacy ACTION node
        label = self.format_action_label(action)
        item = QStandardItem(label)
        item.setData("ACTION", Qt.ItemDataRole.UserRole + 1)
        item.setData(action, Qt.ItemDataRole.UserRole + 2)

        if 'options' in action:
            for i, opt_actions in enumerate(action['options']):
                opt_item = QStandardItem(f"{tr('Option')} {i+1}")
                opt_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
                opt_item.setData({'uid': str(uuid.uuid4())}, Qt.ItemDataRole.UserRole + 2)
                item.appendRow(opt_item)
                for sub_action in opt_actions:
                    sub_item = self._create_action_item(sub_action)
                    opt_item.appendRow(sub_item)
        try:
            self._internalize_item(item)
        except Exception:
            pass
        return item

    def format_action_label(self, action):
        return f"{tr('Action')}: {tr(action.get('type', 'NONE'))}"

    def create_command_item(self, command):
        if 'uid' not in command:
            command['uid'] = str(uuid.uuid4())

        label = self.format_command_label(command)

        # Legacy Warning Handling
        if command.get('legacy_warning'):
             label = f"⚠️ {label}"

        item = QStandardItem(label)
        item.setData("COMMAND", Qt.ItemDataRole.UserRole + 1)
        item.setData(command, Qt.ItemDataRole.UserRole + 2)

        if command.get('legacy_warning'):
             item.setToolTip(f"Legacy Action: {command.get('legacy_original_type', 'Unknown')}\nPlease replace with modern Commands.")
             # item.setForeground(QColor("orange")) # Requires QColor import, relying on label icon for now

        if 'if_true' in command and command['if_true']:
            true_item = QStandardItem(tr("If True"))
            true_item.setData("CMD_BRANCH_TRUE", Qt.ItemDataRole.UserRole + 1)
            true_item.setData({'uid': str(uuid.uuid4())}, Qt.ItemDataRole.UserRole + 2)
            item.appendRow(true_item)
            for child in command['if_true']:
                true_item.appendRow(self.create_command_item(child))

        if 'if_false' in command and command['if_false']:
            false_item = QStandardItem(tr("If False"))
            false_item.setData("CMD_BRANCH_FALSE", Qt.ItemDataRole.UserRole + 1)
            false_item.setData({'uid': str(uuid.uuid4())}, Qt.ItemDataRole.UserRole + 2)
            item.appendRow(false_item)
            for child in command['if_false']:
                false_item.appendRow(self.create_command_item(child))

        # Options (Choice)
        if 'options' in command and command['options']:
            for i, opt_cmds in enumerate(command['options']):
                opt_item = QStandardItem(f"{tr('Option')} {i+1}")
                opt_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
                opt_item.setData({'uid': str(uuid.uuid4())}, Qt.ItemDataRole.UserRole + 2)
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
            card_data = card_item.data(Qt.ItemDataRole.UserRole + 2)
            if card_data and 'id' in card_data:
                try:
                    cid = int(card_data['id'])
                    if cid > max_id: max_id = cid
                except ValueError:
                    pass
        return max_id + 1

    def _find_card_root(self, item):
        """Climb tree to find the CARD node for a given QStandardItem or index."""
        cur_item = item
        # If a QModelIndex was passed, callers should convert to item beforehand.
        while cur_item is not None:
            role = cur_item.data(Qt.ItemDataRole.UserRole + 1)
            if role == 'CARD':
                return cur_item
            parent = cur_item.parent()
            if parent is None:
                return None
            cur_item = parent

    def _update_card_from_child(self, child_item):
        """Reconstruct card data from tree and set it on the CARD node's stored dict."""
        card_item = self._find_card_root(child_item)
        if card_item is None:
            return None
        try:
            updated = self.reconstruct_card_data(card_item)
            if updated:
                card_item.setData(updated, Qt.ItemDataRole.UserRole + 2)
                return updated
        except Exception:
            pass
        return None

    def _find_child_by_role(self, parent_item, role_string):
        """Helper to find a child item with a specific user role data."""
        for i in range(parent_item.rowCount()):
            child = parent_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == role_string:
                return child
        return None

    def _sync_editor_warnings(self, card_item):
        """Ensure an EDITOR_WARNINGS node exists under the card and populate warnings."""
        card_data = card_item.data(Qt.ItemDataRole.UserRole + 2)
        if not isinstance(card_data, dict):
            return
        warnings_list = card_data.get('_editor_warnings', [])

        # Find or create WARNINGS container
        warn_node = self._find_child_by_role(card_item, 'EDITOR_WARNINGS')
        if not warnings_list:
            # remove existing warnings node if present
            if warn_node is not None:
                for i in range(card_item.rowCount()):
                    if card_item.child(i) is warn_node:
                        card_item.removeRow(i)
                        break
            return

        if warn_node is None:
            warn_node = QStandardItem(tr("Warnings"))
            warn_node.setData('EDITOR_WARNINGS', Qt.ItemDataRole.UserRole + 1)
            warn_node.setData({'uid': str(uuid.uuid4())}, Qt.ItemDataRole.UserRole + 2)
            warn_node.setEditable(False)
            card_item.insertRow(0, warn_node)
        else:
            # Clear previous children
            for i in reversed(range(warn_node.rowCount())):
                warn_node.removeRow(i)

        # Populate warning entries
        for w in warnings_list:
            w_item = QStandardItem(str(w))
            w_item.setData('EDITOR_WARNING', Qt.ItemDataRole.UserRole + 1)
            w_item.setData({'uid': str(uuid.uuid4()), 'text': w}, Qt.ItemDataRole.UserRole + 2)
            w_item.setEditable(False)
            warn_node.appendRow(w_item)
