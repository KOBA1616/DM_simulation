from dm_toolkit.gui.editor.action_converter import ActionConverter


def test_select_number_conversion_with_output_key():
    act = {'type': 'SELECT_NUMBER', 'value1': 10, 'output_value_key': 'chosen_num', 'scope': 'PLAYER_SELF'}
    cmd = ActionConverter.convert(act)
    assert cmd['type'] == 'SELECT_NUMBER'
    assert cmd.get('max') == 10
    assert cmd.get('output_value_key') == 'chosen_num'
    assert cmd.get('target_group') == 'PLAYER_SELF'
    assert cmd.get('legacy_warning') is False
