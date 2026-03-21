from dm_toolkit.gui.editor.validators_shared import ModifierValidator


def test_cost_modifier_stat_scaled_requires_fields():
    # Missing stat_key and per_value should error
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED"
    }
    errors = ModifierValidator.validate(mod)
    assert any('STAT_SCALED' in e or 'stat_key' in e or 'per_value' in e for e in errors), errors


def test_cost_modifier_stat_scaled_accepts_valid_config():
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": "MY_MANA_COUNT",
        "per_value": 1,
        "min_stat": 1,
        "max_reduction": 3
    }
    errors = ModifierValidator.validate(mod)
    assert errors == [], f"Expected no validation errors, got: {errors}"
