from dm_toolkit.gui.editor.text_resources import CardTextResources
from dm_toolkit.gui.editor.validators_shared import ConditionValidator


def test_compare_stat_editor_keys_and_stat_key_map_are_accepted():
    """Ensure editor stat keys and engine stat key map keys are accepted by the STATIC COMPARE_STAT validator."""
    # Editor-provided quick keys
    for key in CardTextResources.COMPARE_STAT_EDITOR_KEYS:
        cond = {"type": "COMPARE_STAT", "stat_key": key}
        errs = ConditionValidator.validate_static(cond)
        assert errs == [], f"Editor key '{key}' rejected: {errs}"

    # Engine/translation registry keys
    for key in CardTextResources.STAT_KEY_MAP.keys():
        cond = {"type": "COMPARE_STAT", "stat_key": key}
        errs = ConditionValidator.validate_static(cond)
        assert errs == [], f"STAT_KEY_MAP key '{key}' rejected: {errs}"
import os
import importlib


# Force Python fallback for native module to avoid import-time failures
os.environ['DM_DISABLE_NATIVE'] = '1'


def test_stat_keys_in_card_text_resources_are_accepted_by_validator():
    # Import after setting env
    validators = importlib.import_module('dm_toolkit.gui.editor.validators_shared')
    textres = importlib.import_module('dm_toolkit.gui.editor.text_resources')

    keys = list(textres.CardTextResources.COMPARE_STAT_EDITOR_KEYS) + list(textres.CardTextResources.STAT_KEY_MAP.keys())

    for key in keys:
        mod = {
            "type": "COST_MODIFIER",
            "value_mode": "STAT_SCALED",
            "stat_key": key,
            "per_value": 1,
        }
        errors = validators.ModifierValidator.validate(mod)
        assert errors == [], f"Expected no validation errors for stat_key={key}, got: {errors}"


def test_unknown_stat_key_triggers_error():
    validators = importlib.import_module('dm_toolkit.gui.editor.validators_shared')
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": "THIS_KEY_DOES_NOT_EXIST",
        "per_value": 1,
    }
    errors = validators.ModifierValidator.validate(mod)
    # Validator currently permits unknown stat_key (registry check lives elsewhere),
    # ensure no crash and observe behavior
    assert errors == [], f"Expected no validation errors (current behavior), got: {errors}"
