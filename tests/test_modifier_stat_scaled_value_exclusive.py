from dm_toolkit.gui.editor.validators_shared import ModifierValidator


def test_stat_scaled_must_not_include_value_field():
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": "CREATURES_PLAYED",
        "per_value": 1,
        "value": 2,
    }

    errors = ModifierValidator.validate(mod)

    assert any("must not set 'value'" in e or "STAT_SCALED" in e for e in errors), f"Unexpected errors: {errors}"
