# -*- coding: utf-8 -*-
"""
Shared validators for Static Abilities and Trigger Effects.
Centralizes validation logic for conditions, filters, and ability data structures.

Separated from individual form validators to enable cross-form consistency checks.
"""

from typing import List, Dict, Any


class ConditionValidator:
    """Validates condition data for both static abilities and trigger effects."""
    
    # Valid condition types for static abilities (time-independent)
    VALID_STATIC_CONDITIONS = {
        "NONE",
        "DURING_YOUR_TURN",
        "DURING_OPPONENT_TURN"
    }
    
    # Valid condition types for trigger effects (supports temporal conditions)
    VALID_TRIGGER_CONDITIONS = {
        "NONE",
        "DURING_YOUR_TURN",
        "DURING_OPPONENT_TURN",
        "OPPONENT_DRAW_COUNT"
    }
    
    @staticmethod
    def validate_for_context(condition: Dict[str, Any], context: str) -> List[str]:
        """
        Validates condition data against ability context.
        
        Args:
            condition: Condition dictionary with 'type' and optional fields.
            context: "STATIC" or "TRIGGER" to determine valid condition types.
        
        Returns:
            List of error messages (empty if valid).
        """
        if not isinstance(condition, dict):
            return ["Condition must be a dictionary"]
        
        cond_type = condition.get('type', 'NONE')
        errors = []
        
        if context == "STATIC":
            valid_types = ConditionValidator.VALID_STATIC_CONDITIONS
            context_name = "Static Ability"
        elif context == "TRIGGER":
            valid_types = ConditionValidator.VALID_TRIGGER_CONDITIONS
            context_name = "Trigger Effect"
        else:
            return [f"Unknown context: {context}"]
        
        # Check if condition type is valid for context
        if cond_type not in valid_types:
            errors.append(
                f"'{cond_type}' is invalid for {context_name}. "
                f"Valid types: {', '.join(valid_types)}"
            )
        
        # Context-specific validation
        if context == "TRIGGER":
            # Trigger-specific: validate OPPONENT_DRAW_COUNT if used
            if cond_type == "OPPONENT_DRAW_COUNT":
                if 'value' not in condition or not isinstance(condition.get('value'), int):
                    errors.append("OPPONENT_DRAW_COUNT condition requires 'value' (int)")
        
        return errors
    
    @staticmethod
    def validate_static(condition: Dict[str, Any]) -> List[str]:
        """Convenience method for static ability context."""
        return ConditionValidator.validate_for_context(condition, "STATIC")
    
    @staticmethod
    def validate_trigger(condition: Dict[str, Any]) -> List[str]:
        """Convenience method for trigger effect context."""
        return ConditionValidator.validate_for_context(condition, "TRIGGER")


class FilterValidator:
    """Validates filter/target specification for both contexts."""
    
    @staticmethod
    def validate(filter_def: Dict[str, Any]) -> List[str]:
        """
        Validates filter definition structure and value ranges.
        
        Args:
            filter_def: Filter dictionary with optional numeric fields.
        
        Returns:
            List of error messages.
        """
        if not isinstance(filter_def, dict):
            return ["Filter must be a dictionary"]
        
        errors = []
        
        # Validate numeric fields: cost, power
        numeric_fields = {
            'min_cost': (0, 99999),
            'max_cost': (0, 99999),
            'min_power': (0, 999999),
            'max_power': (0, 999999)
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in filter_def:
                val = filter_def[field]
                if isinstance(val, int):
                    if not (min_val <= val <= max_val):
                        errors.append(
                            f"{field} out of valid range [{min_val}, {max_val}]: {val}"
                        )
                elif not isinstance(val, dict):
                    # Allow dict (for input_link references)
                    errors.append(f"{field} must be int or variable link, got {type(val)}")
        
        # Validate owner (scope) if present
        if 'owner' in filter_def:
            owner = filter_def['owner']
            if owner not in ['SELF', 'OPPONENT', '']:
                errors.append(f"Invalid owner value: '{owner}'. Valid: SELF, OPPONENT, or empty")
        
        # Validate civilization list
        if 'civilizations' in filter_def:
            civs = filter_def['civilizations']
            if not isinstance(civs, list):
                errors.append(f"'civilizations' must be a list, got {type(civs)}")
        
        # Validate type list
        if 'types' in filter_def:
            types = filter_def['types']
            if not isinstance(types, list):
                errors.append(f"'types' must be a list, got {type(types)}")
        
        # Validate race list
        if 'races' in filter_def:
            races = filter_def['races']
            if not isinstance(races, list):
                errors.append(f"'races' must be a list, got {type(races)}")
        
        # Validate zone list
        if 'zones' in filter_def:
            zones = filter_def['zones']
            if not isinstance(zones, list):
                errors.append(f"'zones' must be a list, got {type(zones)}")
        
        # Validate boolean flags
        bool_flags = ['is_tapped', 'is_blocker', 'is_evolution']
        for flag in bool_flags:
            if flag in filter_def:
                val = filter_def[flag]
                if not isinstance(val, int) or val not in [0, 1]:
                    errors.append(f"'{flag}' must be 0 or 1, got {val}")
        
        return errors


class ModifierValidator:
    """Validates static ability (Modifier) data structure."""
    
    VALID_MODIFIER_TYPES = {
        "NONE",
        "COST_MODIFIER",
        "POWER_MODIFIER",
        "GRANT_KEYWORD",
        "SET_KEYWORD",
        "ADD_RESTRICTION"
    }

    VALID_RESTRICTION_KINDS = {
        "TARGET_RESTRICTION",
        "SPELL_RESTRICTION",
        "TARGET_THIS_CANNOT_SELECT",
        "TARGET_THIS_FORCE_SELECT",
    }
    
    @staticmethod
    def validate(modifier: Dict[str, Any]) -> List[str]:
        """
        Validates static ability structure and required fields.
        
        Args:
            modifier: Modifier dictionary from static_abilities list.
        
        Returns:
            List of error messages.
        """
        errors = []
        
        # Required fields
        if 'type' not in modifier:
            errors.append("Modifier must have 'type' field")
            return errors  # Can't validate further without type
        
        mtype = modifier.get('type')
        
        # Validate type value
        if mtype not in ModifierValidator.VALID_MODIFIER_TYPES:
            errors.append(f"Invalid modifier type: '{mtype}'")
        
        # Type-specific validation
        if mtype == "COST_MODIFIER":
            if 'value' not in modifier:
                errors.append("COST_MODIFIER requires 'value' field (cost reduction amount)")
            elif not isinstance(modifier.get('value'), (int, type(None))):
                errors.append(f"COST_MODIFIER 'value' must be int, got {type(modifier.get('value'))}")
        
        elif mtype == "POWER_MODIFIER":
            if 'value' not in modifier:
                errors.append("POWER_MODIFIER requires 'value' field (power adjustment)")
            elif not isinstance(modifier.get('value'), (int, type(None))):
                errors.append(f"POWER_MODIFIER 'value' must be int, got {type(modifier.get('value'))}")
        
        elif mtype in ["GRANT_KEYWORD", "SET_KEYWORD"]:
            # Check mutation_kind first (preferred), fallback to str_val (legacy)
            has_mutation_kind = 'mutation_kind' in modifier and modifier.get('mutation_kind')
            has_str_val = 'str_val' in modifier and modifier.get('str_val')
            
            if not has_mutation_kind and not has_str_val:
                errors.append(f"{mtype} requires 'mutation_kind' or 'str_val' (keyword name)")
            
            # Warn if only legacy str_val is present
            if has_str_val and not has_mutation_kind:
                # Note: This is just a warning, not an error (for backward compatibility)
                pass  # Could log: "Consider migrating str_val to mutation_kind"

        elif mtype == "ADD_RESTRICTION":
            has_mutation_kind = 'mutation_kind' in modifier and modifier.get('mutation_kind')
            has_str_val = 'str_val' in modifier and modifier.get('str_val')
            if not has_mutation_kind and not has_str_val:
                errors.append("ADD_RESTRICTION requires 'mutation_kind' or 'str_val' (restriction kind)")
            kind = (modifier.get('mutation_kind') or modifier.get('str_val') or "").strip()
            if kind and kind not in ModifierValidator.VALID_RESTRICTION_KINDS:
                errors.append(
                    f"Invalid restriction kind: '{kind}'. Valid: {sorted(ModifierValidator.VALID_RESTRICTION_KINDS)}"
                )
        
        # Scope validation (using TargetScope)
        from dm_toolkit.consts import TargetScope
        scope = modifier.get('scope', TargetScope.ALL)
        # Normalize legacy values
        scope = TargetScope.normalize(scope)
        if scope not in TargetScope.all_values():
            errors.append(f"Invalid scope: '{scope}'. Valid: {TargetScope.all_values()}")
        
        # Condition validation (for context "STATIC")
        if 'condition' in modifier:
            cond_errors = ConditionValidator.validate_static(modifier.get('condition', {}))
            errors.extend(cond_errors)
        
        # Filter validation
        if 'filter' in modifier:
            filter_errors = FilterValidator.validate(modifier.get('filter', {}))
            errors.extend(filter_errors)
        
        # IMPORTANT: Modifiers should NOT have 'commands' field
        if 'commands' in modifier and modifier.get('commands'):
            errors.append("Static ability (Modifier) should not have 'commands' field")
        
        # IMPORTANT: Modifiers should NOT have 'trigger' field
        if 'trigger' in modifier and modifier.get('trigger') != 'NONE':
            errors.append("Static ability (Modifier) should not have non-NONE 'trigger' field")
        
        return errors


class TriggerEffectValidator:
    """Validates trigger effect data structure."""
    
    VALID_TRIGGER_TYPES = {
        "ON_PLAY", "ON_OTHER_ENTER", "AT_ATTACK", "ON_DESTROY", "AT_END_OF_TURN",
        "AT_END_OF_OPPONENT_TURN", "ON_BLOCK", "ON_ATTACK_FROM_HAND", "TURN_START",
        "S_TRIGGER", "PASSIVE_CONST", "ON_SHIELD_ADD", "AT_BREAK_SHIELD",
        "ON_CAST_SPELL", "ON_OPPONENT_DRAW", "NONE"
    }
    
    @staticmethod
    def validate(effect: Dict[str, Any]) -> List[str]:
        """
        Validates trigger effect structure and required fields.
        
        Args:
            effect: Effect dictionary from triggers/effects list.
        
        Returns:
            List of error messages.
        """
        errors = []
        
        # Required fields
        trigger = effect.get('trigger', 'NONE')
        if trigger not in TriggerEffectValidator.VALID_TRIGGER_TYPES:
            errors.append(
                f"Invalid trigger type: '{trigger}'. "
                f"Valid types: {', '.join(sorted(TriggerEffectValidator.VALID_TRIGGER_TYPES))}"
            )
        
        # Commands validation (required for non-PASSIVE_CONST triggers)
        if trigger != "PASSIVE_CONST":
            commands = effect.get('commands')
            if not commands:
                # Allow empty for now, but warn if effect has no actions
                if not effect.get('actions'):  # Legacy field
                    errors.append(
                        f"Trigger effect with trigger='{trigger}' has no 'commands' or 'actions'"
                    )
        
        # Condition validation (for context "TRIGGER")
        if 'condition' in effect:
            cond_errors = ConditionValidator.validate_trigger(effect.get('condition', {}))
            errors.extend(cond_errors)
        
        # IMPORTANT: Trigger effects should NOT have modifier-specific fields
        forbidden_fields = ['type', 'value', 'str_val', 'scope']
        for field in forbidden_fields:
            if field in effect and effect.get(field) is not None:
                # Allow 'str_val' in specific cases (e.g., REGISTER_DELAYED_EFFECT)
                # but generally warn about presence
                if field == 'str_val' and trigger in ['PASSIVE_CONST']:
                    continue  # OK for PASSIVE_CONST
                
                # For now, just warn; don't error
                # errors.append(f"Trigger effect should not have '{field}' field")
        
        return errors


class AbilityContextValidator:
    """High-level validator that determines ability type and validates accordingly."""
    
    @staticmethod
    def validate_effect_or_modifier(data: Dict[str, Any]) -> tuple[str, List[str]]:
        """
        Determines if data is a trigger effect or static modifier, and validates.
        
        Args:
            data: Dictionary that could be either type.
        
        Returns:
            (ability_type: "TRIGGER" | "STATIC", errors: List[str])
        """
        errors = []
        
        # Determine type by presence of key fields
        has_trigger = 'trigger' in data and data.get('trigger') != 'NONE'
        has_modifier_type = data.get('type') in ModifierValidator.VALID_MODIFIER_TYPES
        
        if has_modifier_type and not has_trigger:
            # It's a static modifier
            ability_type = "STATIC"
            errors = ModifierValidator.validate(data)
        
        elif has_trigger or data.get('trigger') in TriggerEffectValidator.VALID_TRIGGER_TYPES:
            # It's a trigger effect
            ability_type = "TRIGGER"
            errors = TriggerEffectValidator.validate(data)
        
        else:
            # Ambiguous or invalid
            ability_type = "UNKNOWN"
            errors = [
                "Cannot determine ability type. "
                "Static abilities need 'type' in VALID_MODIFIER_TYPES. "
                "Trigger effects need 'trigger' field."
            ]
        
        return ability_type, errors
    
    @staticmethod
    def describe_ability(data: Dict[str, Any]) -> str:
        """Returns human-readable description of ability type."""
        ability_type, _ = AbilityContextValidator.validate_effect_or_modifier(data)
        return ability_type
