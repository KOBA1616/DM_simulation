from dm_toolkit.gui.editor.validators_shared import ModifierValidator
from dm_toolkit.gui.editor.text_resources import CardTextResources


def test_modifier_validator_accepts_known_stat_key():
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": CardTextResources.COMPARE_STAT_EDITOR_KEYS[0],
        "per_value": 1,
    }
    errs = ModifierValidator.validate(mod)
    assert errs == [], f"Known stat_key unexpectedly rejected: {errs}"


def test_modifier_validator_rejects_unknown_stat_key():
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": "UNKNOWN_STAT_KEY_XXX",
        "per_value": 1,
    }
    errs = ModifierValidator.validate(mod)
    assert any('not a known stat key' in e for e in errs), f"Unknown stat_key did not produce expected error: {errs}"
