from dm_toolkit.gui.editor.models import CommandModel


def test_command_serialization_no_legacy_keys():
    cmd = CommandModel(type='DRAW_CARD')
    cmd.input_var = 'in_key'
    cmd.output_var = 'out_key'
    out = cmd.model_dump()
    # Legacy keys should not be present
    assert 'input_link' not in out
    assert 'input_value_key' in out
    assert out['input_value_key'] == 'in_key'
    assert 'output_link' not in out
    assert 'output_value_key' in out
    assert out['output_value_key'] == 'out_key'


def test_command_serialization_preserves_input_usage_fields():
    cmd = CommandModel.model_validate({
        'type': 'DESTROY',
        'input_value_key': 'ref_power',
        'input_value_usage': 'MAX_POWER',
    })
    out = cmd.model_dump()
    assert out.get('input_value_key') == 'ref_power'
    assert out.get('input_value_usage') == 'MAX_POWER'
