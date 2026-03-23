import sys
import types
import importlib
import os

# Ensure native extension is disabled for tests to avoid import-time native errors
os.environ['DM_DISABLE_NATIVE'] = '1'


# Inject lightweight mocks to satisfy imports that would otherwise load dm_ai_module
mock_i18n = types.SimpleNamespace(tr=lambda *args: ", ".join(args))
sys.modules['dm_toolkit.gui.i18n'] = mock_i18n

mock_models = types.ModuleType('dm_toolkit.gui.editor.models')
class FilterSpec:
    pass

def dict_to_filterspec(d):
    return d

def filterspec_to_dict(fs):
    return fs

mock_models.FilterSpec = FilterSpec
mock_models.dict_to_filterspec = dict_to_filterspec
mock_models.filterspec_to_dict = filterspec_to_dict
sys.modules['dm_toolkit.gui.editor.models'] = mock_models


def test_modifier_validator_requires_stat_scaled_fields():
    # Import after injecting mocks
    validators = importlib.import_module('dm_toolkit.gui.editor.validators_shared')
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED"
    }
    errors = validators.ModifierValidator.validate(mod)
    assert any('STAT_SCALED' in e or 'stat_key' in e or 'per_value' in e for e in errors), errors


def test_modifier_validator_accepts_valid_stat_scaled():
    validators = importlib.import_module('dm_toolkit.gui.editor.validators_shared')
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": "MY_MANA_COUNT",
        "per_value": 1,
        "min_stat": 1,
        "max_reduction": 3
    }
    errors = validators.ModifierValidator.validate(mod)
    assert errors == [], f"Expected no validation errors, got: {errors}"
