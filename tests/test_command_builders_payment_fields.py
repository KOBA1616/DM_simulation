import pytest

from dm_toolkit.command_builders import build_play_card_command


def test_build_play_card_includes_payment_fields_dict():
    cmd = build_play_card_command(card_id=1, source_instance_id=42,
                                  from_zone='HAND', to_zone='BATTLE',
                                  native=False,
                                  payment_mode='ACTIVE_PAYMENT',
                                  reduction_id='r-123',
                                  payment_units=2)

    assert isinstance(cmd, dict)
    assert cmd['type'] == 'PLAY_FROM_ZONE'
    assert cmd['instance_id'] == 42
    assert cmd.get('payment_mode') == 'ACTIVE_PAYMENT'
    assert cmd.get('reduction_id') == 'r-123'
    assert cmd.get('payment_units') == 2
