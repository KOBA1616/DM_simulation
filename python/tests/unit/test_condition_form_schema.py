from dm_toolkit.gui.editor import schema_config


def test_condition_form_schema_basic_keys():
    # Known mappings
    assert schema_config.get_condition_form_fields('OPPONENT_DRAW_COUNT') == ['value']
    assert schema_config.get_condition_form_fields('COMPARE_STAT') == ['stat_key', 'op', 'value']
    assert schema_config.get_condition_form_fields('COMPARE_INPUT') == ['input_value_key', 'op', 'value']


def test_condition_form_schema_unknown_fallback():
    # Unknown condition types should return empty list
    assert schema_config.get_condition_form_fields('SOME_UNKNOWN_COND') == []
    # CUSTOM defined as string-param fallback
    assert schema_config.get_condition_form_fields('CUSTOM') == ['str_val']
