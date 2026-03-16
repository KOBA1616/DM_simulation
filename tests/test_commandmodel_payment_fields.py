from dm_toolkit.gui.editor.models import CommandModel


def test_commandmodel_payment_fields_roundtrip():
    cmd = CommandModel(type='PLAY_FROM_ZONE', payment_mode='ACTIVE_PAYMENT', reduction_id='r_xyz', payment_units=2)
    dumped = cmd.model_dump()
    assert dumped.get('payment_mode') == 'ACTIVE_PAYMENT'
    assert dumped.get('reduction_id') == 'r_xyz'
    assert dumped.get('payment_units') == 2

    # Simulate ingest: ensure ingest_legacy_structure preserves fields when creating new instance
    raw = {'type': 'PLAY_FROM_ZONE', 'payment_mode': 'ACTIVE_PAYMENT', 'reduction_id': 'r_abc', 'payment_units': 3}
    cmd2 = CommandModel.model_validate(raw)
    assert cmd2.payment_mode == 'ACTIVE_PAYMENT'
    assert cmd2.reduction_id == 'r_abc'
    assert cmd2.payment_units == 3
