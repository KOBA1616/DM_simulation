# -*- coding: utf-8 -*-
"""
Tests for validators_shared module.
Validates condition, filter, and ability structure validation logic.
"""

import pytest
from dm_toolkit.gui.editor.validators_shared import (
    ConditionValidator,
    FilterValidator,
    ModifierValidator,
    TriggerEffectValidator,
    AbilityContextValidator
)


class TestConditionValidator:
    """Tests for ConditionValidator."""
    
    def test_valid_static_condition_none(self):
        """NONE condition is valid for static."""
        errors = ConditionValidator.validate_static({"type": "NONE"})
        assert len(errors) == 0
    
    def test_valid_static_condition_during_your_turn(self):
        """DURING_YOUR_TURN is valid for static."""
        errors = ConditionValidator.validate_static({"type": "DURING_YOUR_TURN"})
        assert len(errors) == 0
    
    def test_invalid_static_condition_opponent_draw_count(self):
        """OPPONENT_DRAW_COUNT is invalid for static."""
        errors = ConditionValidator.validate_static({"type": "OPPONENT_DRAW_COUNT"})
        assert len(errors) > 0
        assert "OPPONENT_DRAW_COUNT" in errors[0]
    
    def test_valid_trigger_condition_none(self):
        """NONE condition is valid for trigger."""
        errors = ConditionValidator.validate_trigger({"type": "NONE"})
        assert len(errors) == 0
    
    def test_valid_trigger_condition_opponent_draw_count(self):
        """OPPONENT_DRAW_COUNT is valid for trigger with value."""
        errors = ConditionValidator.validate_trigger({
            "type": "OPPONENT_DRAW_COUNT",
            "value": 3
        })
        assert len(errors) == 0
    
    def test_invalid_trigger_opponent_draw_count_missing_value(self):
        """OPPONENT_DRAW_COUNT requires 'value' field."""
        errors = ConditionValidator.validate_trigger({"type": "OPPONENT_DRAW_COUNT"})
        assert len(errors) > 0
        assert "value" in errors[0].lower()
    
    def test_condition_non_dict(self):
        """Non-dict condition is invalid."""
        errors = ConditionValidator.validate_static("not a dict")
        assert len(errors) > 0


class TestFilterValidator:
    """Tests for FilterValidator."""
    
    def test_empty_filter_valid(self):
        """Empty filter dict is valid."""
        errors = FilterValidator.validate({})
        assert len(errors) == 0
    
    def test_valid_cost_range(self):
        """Valid cost range passes."""
        errors = FilterValidator.validate({
            "min_cost": 0,
            "max_cost": 5
        })
        assert len(errors) == 0
    
    def test_invalid_cost_range(self):
        """Out-of-range cost fails."""
        errors = FilterValidator.validate({
            "min_cost": -1,
            "max_cost": 5
        })
        assert len(errors) > 0
        assert "min_cost" in errors[0]
    
    def test_valid_power_range(self):
        """Valid power range passes."""
        errors = FilterValidator.validate({
            "min_power": 0,
            "max_power": 1000
        })
        assert len(errors) == 0
    
    def test_valid_owner(self):
        """Valid owner values."""
        for owner in ['SELF', 'OPPONENT', '']:
            errors = FilterValidator.validate({"owner": owner})
            assert len(errors) == 0
    
    def test_invalid_owner(self):
        """Invalid owner value."""
        errors = FilterValidator.validate({"owner": "INVALID"})
        assert len(errors) > 0
    
    def test_valid_civilizations_list(self):
        """Valid civilizations list."""
        errors = FilterValidator.validate({
            "civilizations": ["LIGHT", "WATER"]
        })
        assert len(errors) == 0
    
    def test_invalid_civilizations_not_list(self):
        """civilizations must be list."""
        errors = FilterValidator.validate({
            "civilizations": "LIGHT"
        })
        assert len(errors) > 0
    
    def test_valid_boolean_flags(self):
        """Valid boolean flags (0, 1)."""
        errors = FilterValidator.validate({
            "is_tapped": 1,
            "is_blocker": 0,
            "is_evolution": 1
        })
        assert len(errors) == 0
    
    def test_invalid_boolean_flags(self):
        """Boolean flags must be 0 or 1."""
        errors = FilterValidator.validate({
            "is_tapped": 2
        })
        assert len(errors) > 0


class TestModifierValidator:
    """Tests for ModifierValidator."""
    
    def test_valid_cost_modifier(self):
        """Valid COST_MODIFIER structure."""
        errors = ModifierValidator.validate({
            "type": "COST_MODIFIER",
            "value": -2,
            "scope": "ALL",
            "condition": {"type": "NONE"},
            "filter": {}
        })
        assert len(errors) == 0
    
    def test_missing_type(self):
        """Modifier without 'type' fails."""
        errors = ModifierValidator.validate({})
        assert len(errors) > 0
        assert "type" in errors[0].lower()
    
    def test_cost_modifier_missing_value(self):
        """COST_MODIFIER requires 'value'."""
        errors = ModifierValidator.validate({
            "type": "COST_MODIFIER",
            "scope": "ALL"
        })
        assert len(errors) > 0
        assert "value" in errors[0].lower()
    
    def test_grant_keyword_missing_str_val(self):
        """GRANT_KEYWORD requires 'mutation_kind' or 'str_val'."""
        errors = ModifierValidator.validate({
            "type": "GRANT_KEYWORD",
            "scope": "ALL"
        })
        assert len(errors) > 0
        assert ("mutation_kind" in errors[0].lower()) or ("str_val" in errors[0].lower())

    def test_add_restriction_requires_kind(self):
        """ADD_RESTRICTION requires kind in mutation_kind/str_val."""
        errors = ModifierValidator.validate({
            "type": "ADD_RESTRICTION",
            "scope": "ALL"
        })
        assert len(errors) > 0
        assert ("mutation_kind" in errors[0].lower()) or ("str_val" in errors[0].lower())

    def test_add_restriction_valid_kind(self):
        """Valid ADD_RESTRICTION structure."""
        errors = ModifierValidator.validate({
            "type": "ADD_RESTRICTION",
            "mutation_kind": "TARGET_THIS_CANNOT_SELECT",
            "scope": "ALL",
            "condition": {"type": "NONE"},
            "filter": {}
        })
        assert len(errors) == 0
    
    def test_invalid_scope(self):
        """Invalid scope value."""
        errors = ModifierValidator.validate({
            "type": "COST_MODIFIER",
            "value": -1,
            "scope": "INVALID"
        })
        assert len(errors) > 0
        assert "scope" in errors[0].lower()
    
    def test_modifier_with_commands_field_warns(self):
        """Modifier with 'commands' field should error."""
        errors = ModifierValidator.validate({
            "type": "COST_MODIFIER",
            "value": -1,
            "scope": "ALL",
            "commands": [{"type": "TRANSITION"}]  # Not allowed
        })
        assert len(errors) > 0
        assert "commands" in errors[0].lower()
    
    def test_modifier_with_non_none_trigger_warns(self):
        """Modifier with non-NONE trigger should error."""
        errors = ModifierValidator.validate({
            "type": "COST_MODIFIER",
            "value": -1,
            "trigger": "ON_PLAY"  # Not allowed
        })
        assert len(errors) > 0
        assert "trigger" in errors[0].lower()
    
    def test_valid_power_modifier(self):
        """Valid POWER_MODIFIER."""
        errors = ModifierValidator.validate({
            "type": "POWER_MODIFIER",
            "value": 2,
            "scope": "SELF",
            "condition": {"type": "DURING_YOUR_TURN"},
            "filter": {"owner": "SELF"}
        })
        assert len(errors) == 0
    
    def test_invalid_static_condition(self):
        """Invalid condition for static propagates error."""
        errors = ModifierValidator.validate({
            "type": "COST_MODIFIER",
            "value": -1,
            "condition": {"type": "OPPONENT_DRAW_COUNT"}  # Invalid for static
        })
        assert any("OPPONENT_DRAW_COUNT" in e for e in errors)


class TestTriggerEffectValidator:
    """Tests for TriggerEffectValidator."""
    
    def test_valid_trigger_effect(self):
        """Valid trigger effect structure."""
        errors = TriggerEffectValidator.validate({
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "commands": [{"type": "TRANSITION"}]
        })
        assert len(errors) == 0
    
    def test_invalid_trigger_type(self):
        """Invalid trigger type fails."""
        errors = TriggerEffectValidator.validate({
            "trigger": "INVALID_TRIGGER"
        })
        assert len(errors) > 0
        assert "INVALID_TRIGGER" in errors[0]
    
    def test_passive_const_without_commands(self):
        """PASSIVE_CONST doesn't require commands."""
        errors = TriggerEffectValidator.validate({
            "trigger": "PASSIVE_CONST",
            "condition": {"type": "NONE"}
        })
        # PASSIVE_CONST is special, allow without commands
        # The validator should not error on missing commands for this
        assert not any("commands" in e.lower() for e in errors)
    
    def test_on_play_without_commands(self):
        """ON_PLAY with no commands warns."""
        errors = TriggerEffectValidator.validate({
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"}
            # No 'commands' field and no 'actions' field
        })
        # This should warn about missing actions/commands
        assert len(errors) > 0
    
    def test_trigger_with_modifier_fields_warns(self):
        """Trigger with modifier-specific fields should warn."""
        # Currently the validator doesn't error on these, but implementation
        # should consider it in future
        errors = TriggerEffectValidator.validate({
            "trigger": "ON_PLAY",
            "commands": [],
            "type": "COST_MODIFIER",  # Should not be here
            "value": -1  # Should not be here
        })
        # Validator currently doesn't strictly prohibit these (allows legacy data)


class TestAbilityContextValidator:
    """Tests for AbilityContextValidator."""
    
    def test_identify_static_ability(self):
        """Identifies static modifier."""
        ability_type, errors = AbilityContextValidator.validate_effect_or_modifier({
            "type": "COST_MODIFIER",
            "value": -1,
            "scope": "ALL"
        })
        assert ability_type == "STATIC"
        assert len(errors) == 0
    
    def test_identify_trigger_effect(self):
        """Identifies trigger effect."""
        ability_type, errors = AbilityContextValidator.validate_effect_or_modifier({
            "trigger": "ON_PLAY",
            "condition": {"type": "NONE"},
            "commands": []
        })
        assert ability_type == "TRIGGER"
        # May have warnings about missing commands
    
    def test_identify_unknown_ability(self):
        """Identifies unknown/ambiguous ability."""
        ability_type, errors = AbilityContextValidator.validate_effect_or_modifier({
            "unknown": "data"
        })
        assert ability_type == "UNKNOWN"
        assert len(errors) > 0
    
    def test_describe_static(self):
        """Describes static ability correctly."""
        desc = AbilityContextValidator.describe_ability({
            "type": "POWER_MODIFIER",
            "value": 2,
            "scope": "SELF"
        })
        assert desc == "STATIC"
    
    def test_describe_trigger(self):
        """Describes trigger effect correctly."""
        desc = AbilityContextValidator.describe_ability({
            "trigger": "AT_END_OF_TURN",
            "commands": []
        })
        assert desc == "TRIGGER"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
