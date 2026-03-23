import sys
import types

# Provide a lightweight stub for dm_toolkit.gui.i18n used during import-time
stub_i18n = types.SimpleNamespace(tr=lambda x: x)
sys.modules.setdefault('dm_toolkit.gui.i18n', stub_i18n)
# Provide a lightweight stub for dm_ai_module to avoid import-time native wiring
import importlib
dm_stub = types.ModuleType('dm_ai_module')
sys.modules.setdefault('dm_ai_module', dm_stub)

from dm_toolkit.gui.editor.validators_shared import ModifierValidator


def test_stat_scaled_per_value_must_be_positive():
    mod = {
        "type": "COST_MODIFIER",
        "value_mode": "STAT_SCALED",
        "stat_key": "CREATURES_PLAYED",
        "per_value": -1,
    }

    errors = ModifierValidator.validate(mod)

    # Expect a specific validation error about per_value positivity
    assert any("per_value" in e and "> 0" in e for e in errors), f"Unexpected errors: {errors}"
