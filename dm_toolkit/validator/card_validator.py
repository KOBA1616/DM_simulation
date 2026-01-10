import json
import os
import re
from typing import Dict, List, Set, Any, Optional

class ValidationResult:
    def __init__(self, valid: bool, errors: List[str]):
        self.valid = valid
        self.errors = errors

class CardValidator:
    def __init__(self):
        self.valid_zones = {
            "BATTLE_ZONE", "MANA_ZONE", "GRAVEYARD", "HAND", "DECK", "SHIELD_ZONE",
            "EFFECT_BUFFER", "DECK_BOTTOM", "NONE"
        }
        self.valid_command_types = {
            "NONE", "TRANSITION", "MUTATE", "FLOW", "QUERY",
            "DRAW_CARD", "DISCARD", "DESTROY", "BOOST_MANA", "TAP", "UNTAP",
            "POWER_MOD", "ADD_KEYWORD", "RETURN_TO_HAND", "BREAK_SHIELD",
            "SEARCH_DECK", "SHIELD_TRIGGER", "MOVE_CARD", "ADD_MANA",
            "SEND_TO_MANA", "PLAYER_MANA_CHARGE", "SEARCH_DECK_BOTTOM",
            "ADD_SHIELD", "SEND_TO_DECK_BOTTOM", "ATTACK_PLAYER",
            "ATTACK_CREATURE", "BLOCK", "RESOLVE_BATTLE", "RESOLVE_PLAY",
            "RESOLVE_EFFECT", "SHUFFLE_DECK", "LOOK_AND_ADD", "MEKRAID",
            "REVEAL_CARDS", "PLAY_FROM_ZONE", "CAST_SPELL", "SUMMON_TOKEN",
            "SHIELD_BURN", "SELECT_NUMBER", "CHOICE", "LOOK_TO_BUFFER",
            "SELECT_FROM_BUFFER", "PLAY_FROM_BUFFER", "MOVE_BUFFER_TO_ZONE",
            "FRIEND_BURST", "REGISTER_DELAYED_EFFECT", "IF", "IF_ELSE", "ELSE"
        }
        self.valid_types = {
            "CREATURE", "SPELL", "EVOLUTION_CREATURE", "CROSS_GEAR",
            "CASTLE", "PSYCHIC_CREATURE", "GR_CREATURE", "TAMASEED"
        }
        self.valid_civs = {
            "NONE", "LIGHT", "WATER", "DARKNESS", "FIRE", "NATURE", "ZERO"
        }
        self.valid_trigger_types = {
            "NONE", "ON_PLAY", "ON_ATTACK", "ON_DESTROY", "S_TRIGGER",
            "TURN_START", "PASSIVE_CONST", "ON_OTHER_ENTER",
            "ON_ATTACK_FROM_HAND", "ON_BLOCK", "AT_BREAK_SHIELD",
            "BEFORE_BREAK_SHIELD", "ON_SHIELD_ADD", "ON_CAST_SPELL",
            "ON_OPPONENT_DRAW"
        }

        # Mapping from TriggerType to CommandTypes that could trigger it
        self.trigger_cause_map = {
            "ON_PLAY": {"RESOLVE_PLAY", "PLAY_FROM_ZONE", "SUMMON_TOKEN", "CAST_SPELL", "PLAY_FROM_BUFFER"},
            "ON_ATTACK": {"ATTACK_PLAYER", "ATTACK_CREATURE"},
            "ON_DESTROY": {"DESTROY", "SHIELD_BURN"}, # Shield burn puts in grave
            "ON_BLOCK": {"BLOCK"},
            "ON_CAST_SPELL": {"CAST_SPELL", "RESOLVE_PLAY"}, # Spells resolved are cast
            "AT_BREAK_SHIELD": {"BREAK_SHIELD"},
            "ON_SHIELD_ADD": {"ADD_SHIELD"}
        }

    def validate_file(self, filepath: str) -> Dict[int, List[str]]:
        """
        Validates a JSON file containing a list of cards.
        Returns a dictionary mapping card ID to a list of error messages.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {-1: [f"JSON Load Error: {str(e)}"]}

        if not isinstance(data, list):
            return {-1: ["Root element must be a list of cards"]}

        all_errors = {}
        for card in data:
            card_id = card.get('id', -1)
            result = self.validate_card(card)
            if not result.valid:
                all_errors[card_id] = result.errors

        return all_errors

    def validate_card(self, card_data: Dict[str, Any]) -> ValidationResult:
        errors = []

        # Basic fields
        if 'id' not in card_data:
            errors.append("Missing 'id'")
        if 'name' not in card_data:
            errors.append("Missing 'name'")
        if 'type' not in card_data:
            errors.append("Missing 'type'")
        elif card_data['type'] not in self.valid_types:
            errors.append(f"Invalid type: {card_data['type']}")

        # Civilizations
        civs = card_data.get('civilizations', [])
        if not isinstance(civs, list):
             errors.append("'civilizations' must be a list")
        else:
            for civ in civs:
                if civ not in self.valid_civs:
                    errors.append(f"Invalid civilization: {civ}")

        # Effects/Triggers
        effects = card_data.get('triggers', card_data.get('effects', []))
        if not isinstance(effects, list):
             errors.append("'triggers'/'effects' must be a list")
        else:
            for i, effect in enumerate(effects):
                effect_errors = self.validate_effect(effect, context_prefix=f"Effect[{i}]")
                errors.extend(effect_errors)

                # Infinite Loop Check
                loop_errors = self._check_infinite_loops(effect, card_data, f"Effect[{i}]")
                errors.extend(loop_errors)

        # Spell Side
        if card_data.get('spell_side'):
             spell_side_result = self.validate_card(card_data['spell_side'])
             if not spell_side_result.valid:
                 errors.extend([f"SpellSide: {e}" for e in spell_side_result.errors])

        return ValidationResult(len(errors) == 0, errors)

    def validate_effect(self, effect_data: Dict[str, Any], context_prefix: str) -> List[str]:
        errors = []

        trigger = effect_data.get('trigger', 'NONE')
        if trigger not in self.valid_trigger_types:
            errors.append(f"{context_prefix}: Invalid trigger type '{trigger}'")

        commands = effect_data.get('commands', [])
        if not isinstance(commands, list):
            errors.append(f"{context_prefix}: 'commands' must be a list")
            return errors

        # Variable Reference Consistency
        available_vars = set()
        available_vars.add("self") # Implicit self reference often used

        for i, cmd in enumerate(commands):
            cmd_prefix = f"{context_prefix}.Command[{i}]"
            cmd_errors, new_vars = self.validate_command(cmd, available_vars, cmd_prefix)
            errors.extend(cmd_errors)
            available_vars.update(new_vars)

        return errors

    def validate_command(self, cmd_data: Dict[str, Any], available_vars: Set[str], context_prefix: str) -> (List[str], Set[str]):
        errors = []
        new_vars = set()

        cmd_type = cmd_data.get('type', 'NONE')
        if cmd_type not in self.valid_command_types:
            errors.append(f"{context_prefix}: Invalid command type '{cmd_type}'")

        # Zone Validation
        from_zone = cmd_data.get('from_zone')
        if from_zone and from_zone not in self.valid_zones:
             errors.append(f"{context_prefix}: Invalid from_zone '{from_zone}'")

        to_zone = cmd_data.get('to_zone')
        if to_zone and to_zone not in self.valid_zones:
             errors.append(f"{context_prefix}: Invalid to_zone '{to_zone}'")

        # Zone Transition Contradiction (Simple Check)
        if from_zone and to_zone and from_zone == to_zone:
             # Moving to same zone is usually redundant
             pass

        # Variable Reference Check
        input_key = cmd_data.get('input_value_key')
        if input_key:
            if input_key not in available_vars and not self._is_special_var(input_key):
                 errors.append(f"{context_prefix}: Reference to undefined variable '{input_key}'. Available: {list(available_vars)}")

        output_key = cmd_data.get('output_value_key')
        if output_key:
            new_vars.add(output_key)

        next_vars = available_vars.copy()
        next_vars.update(new_vars)

        # Recursive checks for nested commands (IF, IF_ELSE, CHOICE, etc.)
        if 'if_true' in cmd_data:
             for j, sub_cmd in enumerate(cmd_data['if_true']):
                 sub_errors, _ = self.validate_command(sub_cmd, next_vars.copy(), f"{context_prefix}.IfTrue[{j}]")
                 errors.extend(sub_errors)

        if 'if_false' in cmd_data:
             for j, sub_cmd in enumerate(cmd_data['if_false']):
                 sub_errors, _ = self.validate_command(sub_cmd, next_vars.copy(), f"{context_prefix}.IfFalse[{j}]")
                 errors.extend(sub_errors)

        return errors, new_vars

    def _is_special_var(self, key: str) -> bool:
        known_context_keys = {
            "target", "targets", "source", "context", "count", "result",
            "selected_cards", "battle_result"
        }
        if key in known_context_keys: return True
        return False

    def _check_infinite_loops(self, effect_data: Dict[str, Any], card_data: Dict[str, Any], context_prefix: str) -> List[str]:
        """
        Heuristic check for potential infinite loops.
        Detects if a trigger executes commands that trigger the same event type.
        """
        errors = []
        trigger = effect_data.get('trigger', 'NONE')

        # Determine effective trigger for Spells (NONE or ON_CAST_SPELL logic)
        card_type = card_data.get('type', 'CREATURE')
        if card_type == 'SPELL' and trigger == 'NONE':
            # Spells execution is implicitly ON_CAST_SPELL
            trigger = "ON_CAST_SPELL"

        if trigger not in self.trigger_cause_map:
            return []

        risky_commands = self.trigger_cause_map[trigger]

        commands = effect_data.get('commands', [])

        # Helper to recursively scan commands
        def scan_commands(cmd_list, prefix):
            local_errors = []
            for i, cmd in enumerate(cmd_list):
                cmd_type = cmd.get('type', 'NONE')
                if cmd_type in risky_commands:
                    # Found a command that causes the same trigger event.
                    # We should check if it targets SELF or similar, but target info is often runtime.
                    # Warning is safer.
                    local_errors.append(f"{prefix}.Command[{i}]: Potential Infinite Loop - Command '{cmd_type}' may re-trigger '{trigger}'")

                # Recurse
                if 'if_true' in cmd:
                    local_errors.extend(scan_commands(cmd['if_true'], f"{prefix}.Command[{i}].IfTrue"))
                if 'if_false' in cmd:
                    local_errors.extend(scan_commands(cmd['if_false'], f"{prefix}.Command[{i}].IfFalse"))
            return local_errors

        errors.extend(scan_commands(commands, context_prefix))
        return errors

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Validate Card JSON files.")
    parser.add_argument("filepath", help="Path to cards.json")
    args = parser.parse_args()

    validator = CardValidator()
    all_errors = validator.validate_file(args.filepath)

    if not all_errors:
        print(f"Validation Successful: {args.filepath}")
        sys.exit(0)
    else:
        print(f"Validation Failed: {len(all_errors)} cards with errors.")
        for card_id, errors in all_errors.items():
            print(f"Card ID {card_id}:")
            for err in errors:
                print(f"  - {err}")
        sys.exit(1)
