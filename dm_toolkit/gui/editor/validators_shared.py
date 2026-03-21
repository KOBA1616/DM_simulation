from typing import Dict, Any, List, Tuple
import uuid

def validate_cost_reductions(data: List[Dict[str, Any]]) -> List[str]:
    """Validate a list of cost_reduction definitions.

    Returns a list of error messages (empty if valid).
    Checks performed:
    - element is a dict
    - known `type` value
    - numeric fields are of correct sign/type
    - ACTIVE_PAYMENT has minimal required fields
    - duplicate ids are detected and reported
    """
    errors: List[str] = []
    if not isinstance(data, list):
        errors.append("cost_reductions must be a list")
        return errors

    seen_ids = set()
    for idx, cr in enumerate(data):
        prefix = f"cost_reductions[{idx}]"
        if not isinstance(cr, dict):
            errors.append(f"{prefix}: must be an object/dict")
            continue

        typ = cr.get("type")
        if typ not in ("PASSIVE", "ACTIVE_PAYMENT"):
            errors.append(f"{prefix}: unknown type '{typ}' (expected PASSIVE or ACTIVE_PAYMENT)")

        # id checks: id is required and must be a non-empty unique string
        cr_id = cr.get("id")
        if cr_id is None:
            errors.append(f"{prefix}: id is required for cost_reduction entries")
        else:
            if not isinstance(cr_id, str) or not cr_id:
                errors.append(f"{prefix}: id must be a non-empty string")
            elif cr_id in seen_ids:
                errors.append(f"{prefix}: duplicate id '{cr_id}'")
            else:
                seen_ids.add(cr_id)

        # numeric fields
        if "min_mana_cost" in cr:
            mm = cr["min_mana_cost"]
            if not isinstance(mm, int) or mm < 0:
                errors.append(f"{prefix}: min_mana_cost must be integer >= 0")

        if "max_units" in cr:
            mu = cr["max_units"]
            if not isinstance(mu, int) or mu <= 0:
                errors.append(f"{prefix}: max_units must be integer > 0")

        if "reduction_per_unit" in cr:
            rpu = cr["reduction_per_unit"]
            if not isinstance(rpu, int) or rpu <= 0:
                errors.append(f"{prefix}: reduction_per_unit must be integer > 0")

        # ACTIVE_PAYMENT needs at least one of max_units/reduction_per_unit or units
        if typ == "ACTIVE_PAYMENT":
            has_units = "units" in cr and isinstance(cr.get("units"), int) and cr.get("units") > 0
            has_cfg = ("max_units" in cr and isinstance(cr.get("max_units"), int) and cr.get("max_units") > 0) or (
                "reduction_per_unit" in cr and isinstance(cr.get("reduction_per_unit"), int) and cr.get("reduction_per_unit") > 0
            )
            if not (has_units or has_cfg):
                errors.append(f"{prefix}: ACTIVE_PAYMENT requires 'units' or both 'max_units'/'reduction_per_unit' configured")

    return errors


def validate_cost_reduction_item(cr: Dict[str, Any]) -> List[str]:
    """Validate a single cost_reduction dict. Kept for convenience/tests."""
    return validate_cost_reductions([cr])


def generate_missing_ids(cost_reductions: List[Dict[str, Any]]) -> None:
    """Mutate the provided list of cost_reductions, assigning a unique `id` where missing.

    - Uses UUID4 hex strings for IDs.
    - Preserves existing `id` values.
    - Ensures resulting ids are unique within the list by regenerating on collision.
    """
    if not isinstance(cost_reductions, list):
        return

    seen = set()
    # Collect existing ids
    for cr in cost_reductions:
        if isinstance(cr, dict):
            cid = cr.get("id")
            if isinstance(cid, str) and cid:
                seen.add(cid)

    for cr in cost_reductions:
        if not isinstance(cr, dict):
            continue
        if not cr.get("id"):
            # generate until unique
            new_id = uuid.uuid4().hex
            while new_id in seen:
                new_id = uuid.uuid4().hex
            cr["id"] = new_id
            seen.add(new_id)


def detect_passive_static_conflicts(card: Dict[str, Any]) -> List[str]:
    """Detect potential conflicts between PASSIVE cost_reductions and static COST_MODIFIERs.

    Returns a list of warning messages. The rule implemented here is conservative:
    - If the card defines any `cost_reductions` of type PASSIVE and also has any
      `static_abilities` with type `COST_MODIFIER`, emit a warning because the
      effective composition and priority is implementation-defined and may lead
      to surprising behavior at runtime.

    This function intentionally avoids deep semantic overlap analysis; it is
    intended as an early-warning for card authors and editors.
    """
    warnings: List[str] = []
    if not isinstance(card, dict):
        return warnings

    crs = card.get('cost_reductions') or []
    statics = card.get('static_abilities') or []

    has_passive = any(isinstance(cr, dict) and cr.get('type') == 'PASSIVE' for cr in crs)
    has_cost_mod = any(isinstance(m, dict) and m.get('type') == 'COST_MODIFIER' for m in statics)

    if has_passive and has_cost_mod:
        warnings.append(
            "Card defines both PASSIVE cost_reductions and static_abilities of type COST_MODIFIER; "
            "these may overlap or double-apply. Consider consolidating into static_abilities or documenting "
            "priority rules."
        )

    return warnings
# -*- coding: utf-8 -*-
"""
Shared validators for Static Abilities and Trigger Effects.
Centralizes validation logic for conditions, filters, and ability data structures.

Separated from individual form validators to enable cross-form consistency checks.
"""

from typing import List, Dict, Any
from dm_toolkit.gui.i18n import tr
from dm_toolkit.gui.editor.models import FilterSpec, filterspec_to_dict, dict_to_filterspec
from dm_toolkit.gui.editor.text_resources import CardTextResources


class ConditionValidator:
    """Validates condition data for both static abilities and trigger effects."""
    
    # Valid condition types for static abilities (time-independent)
    VALID_STATIC_CONDITIONS = {
        "NONE",
        "DURING_YOUR_TURN",
        "DURING_OPPONENT_TURN",
        "COMPARE_STAT",
        "CARDS_MATCHING_FILTER"
    }
    
    # Valid condition types for trigger effects (supports temporal conditions)
    VALID_TRIGGER_CONDITIONS = {
        "NONE",
        "DURING_YOUR_TURN",
        "DURING_OPPONENT_TURN",
        "OPPONENT_DRAW_COUNT",
        "COMPARE_STAT"
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
                f"Valid types: {tr(', ').join(valid_types)}"
            )
        
        # Context-specific validation
        if context == "TRIGGER":
            # Trigger-specific: validate OPPONENT_DRAW_COUNT if used
            if cond_type == "OPPONENT_DRAW_COUNT":
                if 'value' not in condition or not isinstance(condition.get('value'), int):
                    errors.append("OPPONENT_DRAW_COUNT condition requires 'value' (int)")
            # Validate COMPARE_STAT stat_key against canonical registry
            if cond_type == "COMPARE_STAT":
                stat_key = condition.get('stat_key')
                if not stat_key or not isinstance(stat_key, str):
                    errors.append("COMPARE_STAT requires 'stat_key' (string)")
                else:
                    # Accept if key is in canonical editor list or has a translation entry
                    if (stat_key not in CardTextResources.COMPARE_STAT_EDITOR_KEYS
                            and stat_key not in CardTextResources.STAT_KEY_MAP):
                        errors.append(f"COMPARE_STAT.stat_key '{stat_key}' is not a known stat key")

        # Allow some additional validations for STATIC context as well
        if context == "STATIC":
            # STATIC may also use COMPARE_STAT in the new scheme
            if cond_type == "COMPARE_STAT":
                stat_key = condition.get('stat_key')
                if not stat_key or not isinstance(stat_key, str):
                    errors.append("COMPARE_STAT requires 'stat_key' (string)")
                else:
                    if (stat_key not in CardTextResources.COMPARE_STAT_EDITOR_KEYS
                            and stat_key not in CardTextResources.STAT_KEY_MAP):
                        errors.append(f"COMPARE_STAT.stat_key '{stat_key}' is not a known stat key")

            # STATIC may allow CARDS_MATCHING_FILTER to express existence checks
            if cond_type == "CARDS_MATCHING_FILTER":
                # require op and value and a filter specification
                if 'op' not in condition or not isinstance(condition.get('op'), str):
                    errors.append("CARDS_MATCHING_FILTER requires 'op' (string, e.g. '>=')")
                if 'value' not in condition or not isinstance(condition.get('value'), int):
                    errors.append("CARDS_MATCHING_FILTER requires 'value' (int)")
                if 'filter' not in condition:
                    errors.append("CARDS_MATCHING_FILTER requires 'filter' definition")
                else:
                    # Validate filter structure using FilterValidator
                    try:
                        filter_errors = FilterValidator.validate(condition.get('filter'))
                        errors.extend(filter_errors)
                    except Exception:
                        errors.append("CARDS_MATCHING_FILTER.filter is invalid or could not be parsed")
        
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
        # Normalize to FilterSpec for consistent validation
        if isinstance(filter_def, FilterSpec):
            fs = filter_def
        elif isinstance(filter_def, dict):
            try:
                fs = dict_to_filterspec(filter_def)
            except Exception:
                return ["Failed to convert dict to FilterSpec"]
        else:
            return ["Filter must be a dictionary or FilterSpec"]

        errors: List[str] = []

        # Validate numeric fields: cost, power
        numeric_checks = [
            ('min_cost', 0, 99999),
            ('max_cost', 0, 99999),
            ('min_power', 0, 999999),
            ('max_power', 0, 999999)
        ]

        for field, min_val, max_val in numeric_checks:
            val = getattr(fs, field, None)
            if val is not None:
                if isinstance(val, int):
                    if not (min_val <= val <= max_val):
                        errors.append(f"{field} out of valid range [{min_val}, {max_val}]: {val}")
                else:
                    errors.append(f"{field} must be int when present, got {type(val)}")

        # Validate owner (scope) if present
        owner = getattr(fs, 'owner', None)
        if owner is not None:
            if owner not in ['SELF', 'OPPONENT', '']:
                errors.append(f"Invalid owner value: '{owner}'. Valid: SELF, OPPONENT, or empty")

        # Validate list fields
        for name in ('civilizations', 'types', 'races', 'zones'):
            val = getattr(fs, name, None)
            if val is not None and not isinstance(val, list):
                errors.append(f"'{name}' must be a list, got {type(val)}")

        # Validate boolean flags
        for flag in ('is_tapped', 'is_blocker', 'is_evolution'):
            val = getattr(fs, flag, None)
            if val is not None and not isinstance(val, bool):
                errors.append(f"'{flag}' must be bool when present, got {type(val)}")

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
            # Support new value_mode: FIXED (default) or STAT_SCALED
            value_mode = modifier.get('value_mode', 'FIXED')
            if value_mode not in ('FIXED', 'STAT_SCALED'):
                errors.append("COST_MODIFIER 'value_mode' must be 'FIXED' or 'STAT_SCALED'")

            if value_mode == 'FIXED':
                if 'value' not in modifier:
                    errors.append("COST_MODIFIER with value_mode=FIXED requires 'value' field (cost reduction amount)")
                elif not isinstance(modifier.get('value'), (int, type(None))):
                    errors.append(f"COST_MODIFIER 'value' must be int, got {type(modifier.get('value'))}")

            elif value_mode == 'STAT_SCALED':
                # require stat_key and per_value
                if 'stat_key' not in modifier or not isinstance(modifier.get('stat_key'), str):
                    errors.append("Save blocked: COST_MODIFIER (STAT_SCALED) requires 'stat_key' (string). Example: 'CREATURES_PLAYED'")
                if 'per_value' not in modifier or not isinstance(modifier.get('per_value'), int):
                    errors.append("Save blocked: COST_MODIFIER (STAT_SCALED) requires 'per_value' (int). Example: 1")
                # optional numeric clamps
                if 'min_stat' in modifier and not isinstance(modifier.get('min_stat'), int):
                    errors.append("COST_MODIFIER 'min_stat' must be int when present")
                if 'max_reduction' in modifier and not isinstance(modifier.get('max_reduction'), int):
                    errors.append("COST_MODIFIER 'max_reduction' must be int when present")
        
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
        # 再発防止: static対象条件は target_filter を正とし、legacy filter は互換入力として扱う。
        filter_def = modifier.get('target_filter', modifier.get('filter'))
        if filter_def is not None:
            filter_errors = FilterValidator.validate(filter_def)
            errors.extend(filter_errors)

        if 'target_filter' in modifier and 'filter' in modifier:
            if modifier.get('target_filter') != modifier.get('filter'):
                errors.append("Do not mix different values in 'target_filter' and legacy 'filter'")
        
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
        "AT_END_OF_OPPONENT_TURN", "ON_BLOCK", "TURN_START",
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
                f"Valid types: {tr(', ').join(sorted(TriggerEffectValidator.VALID_TRIGGER_TYPES))}"
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

        # Trigger event conditions should be carried by trigger_filter.
        if 'trigger_filter' in effect:
            filter_errors = FilterValidator.validate(effect.get('trigger_filter', {}))
            errors.extend(filter_errors)

        if 'target_filter' in effect:
            errors.append("Trigger Effect should use 'trigger_filter' instead of 'target_filter'")
        
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
