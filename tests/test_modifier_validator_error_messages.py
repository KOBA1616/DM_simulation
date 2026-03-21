from dm_toolkit.gui.editor.validators_shared import ModifierValidator


def test_stat_scaled_missing_fields_message():
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        # missing stat_key and per_value
    }
    errors = ModifierValidator.validate(mod)
    assert any(e.startswith("Save blocked:") for e in errors), f"Expected save-blocking errors, got: {errors}"
    assert any('stat_key' in e for e in errors)
    assert any('per_value' in e for e in errors)
