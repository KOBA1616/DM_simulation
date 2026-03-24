from dm_toolkit.gui.editor.models import CommandModel


def test_command_serialization_uses_input_output_value_keys():
    cmd = CommandModel(type='DRAW', params={}, input_var='player_hand', output_var='drawn_card')
    dumped = cmd.model_dump()
    # Serializer should emit canonical keys
    assert 'input_value_key' in dumped
    assert 'output_value_key' in dumped
    assert dumped['input_value_key'] == 'player_hand'
    assert dumped['output_value_key'] == 'drawn_card'
    # Should not emit legacy keys
    assert 'input_link' not in dumped
    assert 'input_var' not in dumped
