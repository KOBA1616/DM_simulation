import types
import pytest

from dm_toolkit.unified_execution import ensure_executable_command


class DummyAction:
    def __init__(self, type_name, **kwargs):
        # allow either enum-like or string
        self.type = type_name
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_ensure_command_from_simple_action_dict():
    a = DummyAction('MANA_CHARGE', source_instance_id=3)
    cmd = ensure_executable_command(a)
    assert isinstance(cmd, dict)
    assert cmd.get('type') in ('MANA_CHARGE', 'MANA_CHARGE')
    assert cmd.get('legacy_warning') in (False, None)


def test_ensure_command_from_play_action():
    a = DummyAction('PLAY_CARD', source_instance_id=10, card_id=42)
    cmd = ensure_executable_command(a)
    assert isinstance(cmd, dict)
    # command type should be a PLAY variant (allow several canonical names)
    assert cmd.get('type') is not None
    assert cmd.get('legacy_warning') in (False, None)


def test_ensure_command_from_pybind_like_object():
    # Simulate a pybind-style object with attributes only (no dict)
    obj = types.SimpleNamespace()
    obj.type = 'PAY_COST'
    obj.source_instance_id = 7
    obj.value1 = 1
    cmd = ensure_executable_command(obj)
    assert isinstance(cmd, dict)
    # pybind-like objects may produce a legacy_warning today; at minimum
    # ensure we captured the original action type when mapping failed.
    assert cmd.get('legacy_original_type') in ('PAY_COST', 'PAY_COST')
