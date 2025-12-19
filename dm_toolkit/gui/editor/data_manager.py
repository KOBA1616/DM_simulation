from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt
from dm_toolkit.gui.localization import tr

class CardDataManager:
    """
    Manages data operations for the Card Editor, separating logic from the TreeView.
    Handles loading, saving (reconstruction), and item creation (ID generation).
    """

    def __init__(self, model: QStandardItemModel):
        self.model = model

    def load_data(self, cards_data):
        self.model.clear()
        self.model.setHorizontalHeaderLabels(["Logic Tree"])

        for card_idx, card in enumerate(cards_data):
            card_item = self._create_card_item(card)

            # 1. Add Creature Effects (Triggers)
            # Support both 'triggers' and legacy 'effects' keys
            triggers = card.get('triggers', [])
            if not triggers:
                triggers = card.get('effects', [])

            for eff_idx, effect in enumerate(triggers):
                eff_item = self._create_effect_item(effect)

                # Load Legacy Actions
                for act_idx, action in enumerate(effect.get('actions', [])):
                    act_item = self._create_action_item(action)
                    eff_item.appendRow(act_item)

                # Load Commands
                for cmd_idx, command in enumerate(effect.get('commands', [])):
                    cmd_item = self._create_command_item(command)
                    eff_item.appendRow(cmd_item)

                card_item.appendRow(eff_item)

            # 1.5 Add Static Abilities
            for mod_idx, modifier in enumerate(card.get('static_abilities', [])):
                 mod_item = self._create_modifier_item(modifier)
                 card_item.appendRow(mod_item)

            # 2. Add Reaction Abilities
            for ra_idx, ra in enumerate(card.get('reaction_abilities', [])):
                ra_item = self._create_reaction_item(ra)
                card_item.appendRow(ra_item)

            # 3. Add Spell Side if exists
            spell_side_data = card.get('spell_side')
            if spell_side_data:
                spell_item = self._create_spell_side_item(spell_side_data)
                # Add Spell Effects
                for eff_idx, effect in enumerate(spell_side_data.get('effects', [])):
                    eff_item = self._create_effect_item(effect)

                    # Load Legacy Actions
                    for act_idx, action in enumerate(effect.get('actions', [])):
                        act_item = self._create_action_item(action)
                        eff_item.appendRow(act_item)

                    # Load Commands
                    for cmd_idx, command in enumerate(effect.get('commands', [])):
                        cmd_item = self._create_command_item(command)
                        eff_item.appendRow(cmd_item)

                    spell_item.appendRow(eff_item)
                card_item.appendRow(spell_item)

            self.model.appendRow(card_item)

    def get_full_data(self):
        """Reconstructs the full JSON list from the tree model."""
        cards = []
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            card_item = root.child(i)
            card_data = self.reconstruct_card_data(card_item)
            if card_data:
                cards.append(card_data)
        return cards

    def reconstruct_card_data(self, card_item):
        """Reconstructs a single card's data from its tree item."""
        card_data = card_item.data(Qt.ItemDataRole.UserRole + 2)
        if not card_data: return None

        new_effects = [] # Now strictly triggers
        new_static = [] # New list for static abilities
        new_reactions = []
        spell_side_dict = None

        # Revolution Change extraction
        rev_change_filter = None
        has_rev_change_action = False

        # Iterate children of CARD node
        for j in range(card_item.rowCount()):
            child_item = card_item.child(j)
            item_type = child_item.data(Qt.ItemDataRole.UserRole + 1)

            if item_type == "EFFECT":
                # Reconstruct Effect (Trigger)
                eff_data = self._reconstruct_effect(child_item)
                # Ensure it's treated as a trigger if ambiguous, though EFFECT usually implies trigger
                new_effects.append(eff_data)

                # Check for Revolution Change Action to extract condition
                for act in eff_data.get('actions', []):
                    if act.get('type') == "REVOLUTION_CHANGE":
                        has_rev_change_action = True
                        rev_change_filter = act.get('filter')

            elif item_type == "MODIFIER":
                # Reconstruct Static Ability
                mod_data = self._reconstruct_modifier(child_item)
                new_static.append(mod_data)

            elif item_type == "REACTION_ABILITY":
                # Reconstruct Reaction Ability
                ra_data = child_item.data(Qt.ItemDataRole.UserRole + 2)
                new_reactions.append(ra_data)

            elif item_type == "SPELL_SIDE":
                # Reconstruct Spell Side
                spell_side_data = child_item.data(Qt.ItemDataRole.UserRole + 2)
                spell_side_effects = []
                for k in range(child_item.rowCount()):
                    eff_item = child_item.child(k)
                    if eff_item.data(Qt.ItemDataRole.UserRole + 1) == "EFFECT":
                        spell_side_effects.append(self._reconstruct_effect(eff_item))

                spell_side_data['effects'] = spell_side_effects
                spell_side_dict = spell_side_data

        card_data['effects'] = new_effects
        card_data['triggers'] = new_effects # Populate both for compatibility/transition
        card_data['static_abilities'] = new_static
        card_data['reaction_abilities'] = new_reactions
        if spell_side_dict:
            card_data['spell_side'] = spell_side_dict
        else:
            if 'spell_side' in card_data:
                del card_data['spell_side']

        # Auto-set Revolution Change Keyword and Condition
        if 'keywords' not in card_data:
            card_data['keywords'] = {}

        if has_rev_change_action and rev_change_filter:
            card_data['keywords']['revolution_change'] = True
            card_data['revolution_change_condition'] = rev_change_filter
        else:
            # If removed from tree, clear from root data
            if 'revolution_change' in card_data['keywords']:
                del card_data['keywords']['revolution_change']
            if 'revolution_change_condition' in card_data:
                del card_data['revolution_change_condition']

        return card_data

    def _reconstruct_effect(self, eff_item):
        eff_data = eff_item.data(Qt.ItemDataRole.UserRole + 2)
        new_actions = []
        new_commands = []

        for k in range(eff_item.rowCount()):
            item = eff_item.child(k)
            item_type = item.data(Qt.ItemDataRole.UserRole + 1)

            if item_type == "ACTION":
                act_data = self._reconstruct_action(item)
                new_actions.append(act_data)
            elif item_type == "COMMAND":
                cmd_data = self._reconstruct_command(item)
                new_commands.append(cmd_data)

        eff_data['actions'] = new_actions

        # Handle commands list
        if new_commands:
            eff_data['commands'] = new_commands
        elif 'commands' in eff_data:
            # If we had commands but deleted all of them, remove the key
            del eff_data['commands']

        return eff_data

    def _reconstruct_action(self, act_item):
        act_data = act_item.data(Qt.ItemDataRole.UserRole + 2)
        # Check for child options (Nested Actions)
        if act_item.rowCount() > 0:
            options = []
            for m in range(act_item.rowCount()):
                option_item = act_item.child(m)
                if option_item.data(Qt.ItemDataRole.UserRole + 1) == "OPTION":
                    # Reconstruct option actions
                    option_actions = []
                    for n in range(option_item.rowCount()):
                        sub_act_item = option_item.child(n)
                        if sub_act_item.data(Qt.ItemDataRole.UserRole + 1) == "ACTION":
                            option_actions.append(self._reconstruct_action(sub_act_item))
                    options.append(option_actions)
            act_data['options'] = options
        elif 'options' in act_data:
            # Clear options if no children exist in the view (removed by user)
            del act_data['options']

        return act_data

    def _reconstruct_modifier(self, mod_item):
        """Reconstructs a modifier dict from the item data."""
        # The item data already contains the updated dictionary from the form
        return mod_item.data(Qt.ItemDataRole.UserRole + 2)

    def _reconstruct_command(self, cmd_item):
        cmd_data = cmd_item.data(Qt.ItemDataRole.UserRole + 2)

        if_true_list = []
        if_false_list = []

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

        if if_true_list:
            cmd_data['if_true'] = if_true_list
        elif 'if_true' in cmd_data:
            del cmd_data['if_true']

        if if_false_list:
            cmd_data['if_false'] = if_false_list
        elif 'if_false' in cmd_data:
            del cmd_data['if_false']

        return cmd_data

    def add_new_card(self):
        new_id = self._generate_new_id()
        # Updated: Use 'civilizations' list by default
        new_card = {
            "id": new_id, "name": "New Card",
            "civilizations": ["FIRE"], "type": "CREATURE",
            "cost": 1, "power": 1000, "races": [], "effects": []
        }
        # Legacy cleanup just in case
        if "civilization" in new_card: del new_card["civilization"]

        item = self._create_card_item(new_card)
        self.model.appendRow(item)
        return item

    def add_child_item(self, parent_index, item_type, data, label):
        if not parent_index.isValid(): return None
        parent_item = self.model.itemFromIndex(parent_index)

        new_item = QStandardItem(label)
        new_item.setData(item_type, Qt.ItemDataRole.UserRole + 1)
        new_item.setData(data, Qt.ItemDataRole.UserRole + 2)

        # For Twinpact structure:
        # If adding EFFECT, MODIFIER, or REACTION_ABILITY to CARD, insert BEFORE 'SPELL_SIDE' if it exists.
        if (item_type == "EFFECT" or item_type == "MODIFIER" or item_type == "REACTION_ABILITY") and parent_item.data(Qt.ItemDataRole.UserRole + 1) == "CARD":
            spell_side_row = -1
            for i in range(parent_item.rowCount()):
                child = parent_item.child(i)
                if child.data(Qt.ItemDataRole.UserRole + 1) == "SPELL_SIDE":
                    spell_side_row = i
                    break

            if spell_side_row != -1:
                parent_item.insertRow(spell_side_row, new_item)
                return new_item

        parent_item.appendRow(new_item)
        return new_item

    def add_spell_side_item(self, card_item):
        """Adds a Spell Side node to the given card item if not present."""
        # Check if exists
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "SPELL_SIDE":
                return child # Already exists

        # Default Spell Data
        spell_data = {
            "name": "New Spell Side",
            "type": "SPELL",
            "cost": 1,
            "civilizations": [],
            "effects": []
        }
        item = self._create_spell_side_item(spell_data)
        card_item.appendRow(item)
        return item

    def remove_spell_side_item(self, card_item):
        """Removes the Spell Side node from the given card item."""
        for i in reversed(range(card_item.rowCount())):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "SPELL_SIDE":
                card_item.removeRow(i)
                return True
        return False

    def add_revolution_change_logic(self, card_item):
        """Adds generic Revolution Change logic (Effect + Action) to the card."""
        # Check if already exists to avoid duplicates
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "EFFECT":
                eff_data = child.data(Qt.ItemDataRole.UserRole + 2)
                for act in eff_data.get('actions', []):
                    if act.get('type') == 'REVOLUTION_CHANGE':
                        return child # Already exists

        # 1. Create Effect (Trigger: ON_ATTACK_FROM_HAND)
        eff_data = {
            "trigger": "ON_ATTACK_FROM_HAND",
            "condition": {"type": "NONE"},
            "actions": []
        }
        eff_item = self._create_effect_item(eff_data)

        # 2. Create Action (Type: REVOLUTION_CHANGE)
        act_data = {
            "type": "REVOLUTION_CHANGE",
            "filter": {
                "civilizations": ["FIRE"], # Default
                "races": ["Dragon"], # Default
                "min_cost": 5
            }
        }
        act_item = self._create_action_item(act_data)
        eff_item.appendRow(act_item)

        # Add to card (before spell side if any)
        spell_side_row = -1
        for i in range(card_item.rowCount()):
            child = card_item.child(i)
            if child.data(Qt.ItemDataRole.UserRole + 1) == "SPELL_SIDE":
                spell_side_row = i
                break

        if spell_side_row != -1:
            card_item.insertRow(spell_side_row, eff_item)
        else:
            card_item.appendRow(eff_item)

        return eff_item

    def remove_revolution_change_logic(self, card_item):
        """Removes the Revolution Change effect from the card."""
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
        """Adds option slots to an action item."""
        current_options = 0
        for i in range(action_item.rowCount()):
             if action_item.child(i).data(Qt.ItemDataRole.UserRole + 1) == "OPTION":
                  current_options += 1

        for i in range(count):
             opt_num = current_options + i + 1
             opt_item = QStandardItem(f"{tr('Option')} {opt_num}")
             opt_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
             opt_item.setData({}, Qt.ItemDataRole.UserRole + 2)
             action_item.appendRow(opt_item)

    def add_command_branches(self, cmd_item):
        """Adds True/False branches to a command item."""
        # Check if already exists
        has_true = False
        has_false = False
        for i in range(cmd_item.rowCount()):
            role = cmd_item.child(i).data(Qt.ItemDataRole.UserRole + 1)
            if role == "CMD_BRANCH_TRUE": has_true = True
            if role == "CMD_BRANCH_FALSE": has_false = True

        if not has_true:
            true_item = QStandardItem(tr("If True"))
            true_item.setData("CMD_BRANCH_TRUE", Qt.ItemDataRole.UserRole + 1)
            true_item.setData({}, Qt.ItemDataRole.UserRole + 2)
            cmd_item.appendRow(true_item)

        if not has_false:
            false_item = QStandardItem(tr("If False"))
            false_item.setData("CMD_BRANCH_FALSE", Qt.ItemDataRole.UserRole + 1)
            false_item.setData({}, Qt.ItemDataRole.UserRole + 2)
            cmd_item.appendRow(false_item)

    def _create_card_item(self, card):
        item = QStandardItem(f"{card.get('id')} - {card.get('name', 'No Name')}")
        item.setData("CARD", Qt.ItemDataRole.UserRole + 1)
        item.setData(card, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_spell_side_item(self, spell_data):
        item = QStandardItem(f"{tr('Spell Side')}: {spell_data.get('name', 'No Name')}")
        item.setData("SPELL_SIDE", Qt.ItemDataRole.UserRole + 1)
        item.setData(spell_data, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_effect_item(self, effect):
        trig = effect.get('trigger', 'NONE')
        item = QStandardItem(f"{tr('Effect')}: {tr(trig)}")
        item.setData("EFFECT", Qt.ItemDataRole.UserRole + 1)
        item.setData(effect, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_modifier_item(self, modifier):
        mtype = modifier.get('type', 'NONE')
        item = QStandardItem(f"{tr('Static')}: {tr(mtype)}")
        item.setData("MODIFIER", Qt.ItemDataRole.UserRole + 1)
        item.setData(modifier, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_reaction_item(self, reaction):
        rtype = reaction.get('type', 'NONE')
        item = QStandardItem(f"{tr('Reaction Ability')}: {rtype}")
        item.setData("REACTION_ABILITY", Qt.ItemDataRole.UserRole + 1)
        item.setData(reaction, Qt.ItemDataRole.UserRole + 2)
        return item

    def _create_command_item(self, command):
        cmd_type = command.get('type', 'NONE')
        item = QStandardItem(f"{tr('Command')}: {tr(cmd_type)}")
        item.setData("COMMAND", Qt.ItemDataRole.UserRole + 1)
        item.setData(command, Qt.ItemDataRole.UserRole + 2)

        # Recursion for if_true/if_false
        if 'if_true' in command and command['if_true']:
            true_item = QStandardItem(tr("If True"))
            true_item.setData("CMD_BRANCH_TRUE", Qt.ItemDataRole.UserRole + 1)
            true_item.setData({}, Qt.ItemDataRole.UserRole + 2)
            item.appendRow(true_item)
            for child in command['if_true']:
                true_item.appendRow(self._create_command_item(child))

        if 'if_false' in command and command['if_false']:
            false_item = QStandardItem(tr("If False"))
            false_item.setData("CMD_BRANCH_FALSE", Qt.ItemDataRole.UserRole + 1)
            false_item.setData({}, Qt.ItemDataRole.UserRole + 2)
            item.appendRow(false_item)
            for child in command['if_false']:
                false_item.appendRow(self._create_command_item(child))

        return item

    def _create_action_item(self, action):
        act_type = action.get('type', 'NONE')
        display_type = tr(act_type)

        if act_type == "GET_GAME_STAT":
             display_type = f"{tr('Reference')} {tr(action.get('str_val',''))}"
        elif act_type == "APPLY_MODIFIER" and action.get('str_val') == "COST":
             display_type = tr("Reduce Cost by")
             if action.get('input_value_key'):
                 display_type += f" [{action.get('input_value_key')}]"
             else:
                 display_type += f" {action.get('value1', 0)}"
        elif act_type == "COST_REFERENCE":
             display_type = f"{tr('COST_REFERENCE')} ({tr(action.get('str_val',''))})"
        elif act_type in ["SELECT_TARGET", "DESTROY", "RETURN_TO_HAND", "SEND_TO_MANA", "TAP"]:
             count = action.get('filter', {}).get('count')
             if count:
                 display_type += f" (Count: {count})"
        elif act_type == "MOVE_CARD":
             dest = action.get('destination_zone', 'HAND')
             source = action.get('source_zone', 'NONE') # Optional source tracking

             # Contextual Naming based on Source -> Destination
             # Hand -> Graveyard = Discard
             # Battle/Target -> Graveyard = Destroy
             # Shield -> Graveyard = Burn
             # Hand -> Mana = Charge
             # Battle -> Hand = Bounce

             display_type = tr(dest)

             if dest == "MANA_ZONE":
                 display_type = tr("SEND_TO_MANA")
             elif dest == "GRAVEYARD":
                 if source == "HAND":
                     display_type = tr("DISCARD")
                 elif source == "SHIELD_ZONE":
                     display_type = tr("SHIELD_BURN")
                 else:
                     display_type = tr("DESTROY")
             elif dest == "HAND":
                 display_type = tr("RETURN_TO_HAND")
             elif dest == "DECK_BOTTOM":
                 display_type = tr("SEND_TO_DECK_BOTTOM")
             elif dest == "SHIELD_ZONE":
                 display_type = tr("ADD_SHIELD")

             count = action.get('filter', {}).get('count')
             if count:
                 display_type += f" (Count: {count})"

        elif act_type == "REVOLUTION_CHANGE":
             display_type = tr("Revolution Change")
        elif act_type == "SELECT_OPTION":
             display_type = tr("Mode Selection")

        item = QStandardItem(f"{tr('Action')}: {display_type}")
        item.setData("ACTION", Qt.ItemDataRole.UserRole + 1)
        item.setData(action, Qt.ItemDataRole.UserRole + 2)

        # Recursively add options if present
        if 'options' in action:
            for i, opt_actions in enumerate(action['options']):
                opt_item = QStandardItem(f"{tr('Option')} {i+1}")
                opt_item.setData("OPTION", Qt.ItemDataRole.UserRole + 1)
                opt_item.setData({}, Qt.ItemDataRole.UserRole + 2) # Empty data for option container
                item.appendRow(opt_item)
                for sub_action in opt_actions:
                    sub_item = self._create_action_item(sub_action)
                    opt_item.appendRow(sub_item)

        return item

    def _generate_new_id(self):
        max_id = 0
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            card_item = root.child(i)
            card_data = card_item.data(Qt.ItemDataRole.UserRole + 2)
            if card_data and 'id' in card_data:
                try:
                    cid = int(card_data['id'])
                    if cid > max_id:
                        max_id = cid
                except ValueError:
                    pass
        return max_id + 1
