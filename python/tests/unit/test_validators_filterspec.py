from dm_toolkit.gui.editor.validators_shared import FilterValidator
from dm_toolkit.gui.editor.models import FilterSpec


def test_filter_validator_accepts_filterspec_and_detects_invalid_min_cost():
    fs = FilterSpec(min_cost=-5)
    errs = FilterValidator.validate(fs)
    assert any('min_cost' in e for e in errs)
