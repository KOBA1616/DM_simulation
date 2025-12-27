
import pytest
from dm_toolkit.gui.editor.command_model import CommandDef, WarningCommand

def test_command_def_to_dict_flat():
    cmd = CommandDef(
        uid="123",
        type="TEST",
        params={"amount": 1, "custom": "value"}
    )
    d = cmd.to_dict()
    assert d["uid"] == "123"
    assert d["type"] == "TEST"
    assert d["amount"] == 1
    assert d["custom"] == "value"
    assert "params" not in d

def test_command_def_from_dict_flat():
    d = {
        "uid": "456",
        "type": "TEST2",
        "amount": 2,
        "custom": "val2",
        "input_value_key": "in1",
        "options": [[{"type": "SUB"}]]
    }
    cmd = CommandDef.from_dict(d)
    assert cmd.uid == "456"
    assert cmd.type == "TEST2"
    assert cmd.params["amount"] == 2
    assert cmd.params["custom"] == "val2"
    assert cmd.input_value_key == "in1"
    assert cmd.options[0][0]["type"] == "SUB"

def test_warning_command():
    wc = WarningCommand(
        uid="789",
        warning="Test warning",
        original_action={"type": "LEGACY"}
    )
    d = wc.to_dict()
    assert d["legacy_warning"] is True
    assert d["warning"] == "Test warning"
    assert d["legacy_original_action"]["type"] == "LEGACY"
    assert d["uid"] == "789"

    wc2 = WarningCommand.from_dict(d)
    assert wc2.warning == "Test warning"
    assert wc2.uid == "789"
