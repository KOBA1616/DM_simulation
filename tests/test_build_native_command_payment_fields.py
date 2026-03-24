from types import SimpleNamespace

import dm_toolkit.command_builders as cb


class FakeCommandDef:
    def __init__(self):
        # Common fields used by _build_native_command
        self.type = None
        self.instance_id = None
        self.source_instance_id = None
        self.target_instance = None
        self.owner_id = None
        self.amount = None
        self.from_zone = None
        self.to_zone = None
        self.mutation_kind = None
        self.str_param = None
        self.optional = None
        self.up_to = None
        self.input_value_key = None
        self.input_value_usage = None
        self.output_value_key = None
        self.slot_index = None
        self.target_slot_index = None
        self.target_group = None
        self.target_filter = None
        self.condition = None
        self.if_true = []
        self.if_false = []
        self.options = []
        # Payment-specific
        self.payment_mode = None
        self.reduction_id = None
        self.payment_units = None


def test_native_build_maps_payment_fields():
    orig_has_native = getattr(cb, '_HAS_NATIVE', None)
    orig_CommandDef = getattr(cb, '_CommandDef', None)
    orig_CommandType = getattr(cb, '_CommandType', None)
    try:
        cb._HAS_NATIVE = True
        cb._CommandDef = FakeCommandDef
        cb._CommandType = SimpleNamespace(PLAY_FROM_ZONE=999)

        cmd = cb._build_native_command('PLAY_FROM_ZONE', instance_id=42,
                                       payment_mode='ACTIVE_PAYMENT',
                                       reduction_id='r-99',
                                       payment_units=3)

        assert isinstance(cmd, FakeCommandDef)
        assert cmd.type == 999
        assert cmd.instance_id == 42
        assert cmd.payment_mode == 'ACTIVE_PAYMENT'
        assert cmd.reduction_id == 'r-99'
        assert cmd.payment_units == 3
    finally:
        # restore
        cb._HAS_NATIVE = orig_has_native
        cb._CommandDef = orig_CommandDef
        cb._CommandType = orig_CommandType
