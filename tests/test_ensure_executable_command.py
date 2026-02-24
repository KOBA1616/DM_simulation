import json
from dm_toolkit.unified_execution import ensure_executable_command, to_command_dict


def test_dict_action_play_card():
    action = {"type": "PLAY_CARD", "card_id": 123, "source_instance_id": 77}
    cmd = ensure_executable_command(action)
    assert isinstance(cmd, dict)
    # Accept several PLAY-like mappings (engine may normalize to PLAY_FROM_ZONE)
    assert isinstance(cmd.get('type'), str) and 'PLAY' in cmd.get('type')
    # instance id preserved
    assert cmd.get('instance_id') == 77 or cmd.get('source_instance_id') == 77


class AttrAction:
    def __init__(self):
        self.type = 'MANA_CHARGE'
        self.card_id = 5
        self.source_instance_id = 200


def test_attribute_like_action():
    a = AttrAction()
    cmd = ensure_executable_command(a)
    assert isinstance(cmd, dict)
    assert cmd.get('type') == 'MANA_CHARGE'
    assert cmd.get('instance_id') == 200 or cmd.get('source_instance_id') == 200


class JsonAction:
    def to_json(self):
        return {"type": "PASS"}


def test_to_json_action():
    ja = JsonAction()
    cmd = ensure_executable_command(ja)
    assert isinstance(cmd, dict)
    assert cmd.get('type') == 'PASS'

def test_native_command_passthrough():
    try:
        import dm_ai_module
        if not hasattr(dm_ai_module, 'CommandDef'):
             return # Skip if incomplete binding

        cmd = dm_ai_module.CommandDef()
        # Set some properties
        if hasattr(dm_ai_module.CommandType, 'PASS'):
             cmd.type = dm_ai_module.CommandType.PASS

        result = ensure_executable_command(cmd)

        # Verify identity (pass-through)
        assert result is cmd
        # Verify type
        assert isinstance(result, dm_ai_module.CommandDef)

    except ImportError:
        pass # Skip if native module not available
