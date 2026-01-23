import pytest
from dm_toolkit.unified_execution import to_command_dict, ensure_executable_command
from dm_toolkit.action_to_command import map_action

class MockAction:
    def __init__(self, data):
        self.__dict__ = data

class MockCommandDef:
    def __init__(self, data):
        self.data = data
    def to_json(self):
        return self.data

def test_conversion_consistency_basic():
    """Test basic conversion from dict to command."""
    action = {"type": "DRAW_CARD", "value1": 2}

    # 1. to_command_dict
    cmd1 = to_command_dict(action)
    # map_action maps DRAW_CARD to DRAW_CARD (legacy) or TRANSITION?
    # map_action source:
    # elif act_type == "DRAW_CARD":
    #     cmd['type'] = act_type (DRAW_CARD)
    #     _transfer_common_move_fields(act_data, cmd) -> sets amount from value1

    assert cmd1['type'] == "DRAW_CARD"
    assert cmd1['amount'] == 2
    # value1 is NOT guaranteed to be in cmd1 by map_action unless explicitly copied
    # map_action does NOT copy 'value1' explicitly for DRAW_CARD except via _finalize_command which sets amount

    # 2. ensure_executable_command
    cmd2 = ensure_executable_command(action)
    assert cmd2['type'] == "DRAW_CARD"
    assert cmd2['amount'] == 2
    assert cmd2['value1'] == 2  # Normalized
    assert cmd2['legacy_original_type'] == "DRAW_CARD"

def test_conversion_consistency_conflict():
    """Test behavior when input has conflicting legacy and canonical fields."""
    # Input has both, different values.
    # map_action prioritizes 'amount' if present in input for _finalize_command
    action = {"type": "DRAW_CARD", "value1": 10, "amount": 5}

    cmd1 = to_command_dict(action)
    assert cmd1['amount'] == 5
    # value1 is not in cmd1 usually, unless copied.
    # map_action doesn't blindly copy all fields.

    cmd2 = ensure_executable_command(action)
    assert cmd2['amount'] == 5
    assert cmd2['value1'] == 5  # normalize_legacy_fields syncs them.
    # Since amount=5 is in cmd, and value1 is NOT in cmd (dropped by map_action),
    # normalize_legacy_fields sets value1 = amount.

    # What if map_action DID copy value1?
    # Some handlers might copy it.
    # e.g. _handle_specific_moves -> SHIELD_BURN copies value1 explicitly

    action_burn = {"type": "SHIELD_BURN", "value1": 10, "amount": 5}
    # _handle_specific_moves:
    # if act_type == "SHIELD_BURN": cmd['amount'] = act.get('value1', 1)
    # It IGNORES 'amount' in input and uses 'value1' for SHIELD_BURN specific logic!
    # But _finalize_command runs after.
    # _finalize_command:
    # if 'amount' not in cmd: ...
    # So if _handle_specific_moves set 'amount', _finalize_command keeps it.

    cmd_burn = to_command_dict(action_burn)
    assert cmd_burn['type'] == "SHIELD_BURN"
    assert cmd_burn['amount'] == 5 # Now prefers 'amount' if present

    # verify logic in action_to_command.py:
    # elif act_type in [..., "SHIELD_BURN", ...]:
    #     _handle_specific_moves(...)
    #
    # _handle_specific_moves:
    # if act_type == "SHIELD_BURN": cmd['amount'] = act.get('amount') or act.get('value1', 1)

    cmd_burn_exec = ensure_executable_command(action_burn)
    assert cmd_burn_exec['amount'] == 5
    assert cmd_burn_exec['value1'] == 5

def test_ensure_executable_idempotency():
    """Test that ensure_executable_command is idempotent-ish."""
    action = {"type": "DRAW_CARD", "value1": 2}
    cmd = ensure_executable_command(action)

    # Pass the result back in
    cmd2 = ensure_executable_command(cmd)

    assert cmd2['amount'] == 2
    assert cmd2['value1'] == 2
    assert cmd2['type'] == "DRAW_CARD"

def test_object_input():
    """Test conversion from objects."""
    obj = MockAction({"type": "DRAW_CARD", "value1": 3})
    cmd = ensure_executable_command(obj)
    assert cmd['amount'] == 3
    assert cmd['value1'] == 3

def test_missing_fields_normalization():
    """Test that missing fields are populated."""
    # Action with only canonical fields
    action = {"type": "DRAW_CARD", "amount": 4}
    cmd = ensure_executable_command(action)
    assert cmd['value1'] == 4

    # Action with only legacy fields
    action2 = {"type": "DRAW_CARD", "value1": 4}
    cmd2 = ensure_executable_command(action2)
    assert cmd2['amount'] == 4

def test_str_normalization():
    """Test string parameter normalization."""
    action = {"type": "MUTATE", "str_val": "TEST", "target_group": "NONE"}
    cmd = ensure_executable_command(action)
    assert cmd['str_param'] == "TEST"
    assert cmd['str_val'] == "TEST"

    action2 = {"type": "MUTATE", "str_param": "TEST2", "target_group": "NONE"}
    cmd2 = ensure_executable_command(action2)
    assert cmd2['str_val'] == "TEST2"
    assert cmd2['str_param'] == "TEST2"
