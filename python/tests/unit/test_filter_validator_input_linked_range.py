from dm_toolkit.gui.editor.validators_shared import FilterValidator


def test_filter_validator_accepts_input_linked_max_power_dict() -> None:
    filt = {
        'types': ['CREATURE'],
        'max_power': {
            'input_link': 'selected_power',
            'input_value_usage': 'MAX_POWER',
        },
    }
    errors = FilterValidator.validate(filt)
    assert errors == []


def test_filter_validator_rejects_invalid_linked_range_dict() -> None:
    filt = {
        'max_power': {
            'input_value_usage': 'MAX_POWER',
        },
    }
    errors = FilterValidator.validate(filt)
    assert any('max_power dict must include input_link or input_value_key' in e for e in errors)
