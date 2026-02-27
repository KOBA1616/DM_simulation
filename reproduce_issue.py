
import sys
from typing import Any, Dict
from unittest.mock import MagicMock

# Mock dm_ai_module
mock_dm = MagicMock()
class MockCommandDef:
    def __init__(self):
        self.instance_id = 1
        self.owner_id = 1
    def to_dict(self):
        return {"type": "DRAW_CARD", "amount": 1, "instance_id": 1, "owner_id": 1}
    def __str__(self):
        return "MockCommandDef(id=1)"

mock_dm.CommandDef = MockCommandDef
sys.modules['dm_ai_module'] = mock_dm

from dm_toolkit.unified_execution import ensure_executable_command, execute_command
from dm_toolkit.engine.compat import EngineCompat

def test_direct_execution():
    print("Testing direct execution flow...")
    cmd_obj = MockCommandDef()
    state = MagicMock()
    state.active_player_id = 1

    # Mock CommandSystem
    mock_dm.CommandSystem = MagicMock()

    # Execute
    execute_command(state, cmd_obj)

    # Verify CommandSystem.execute_command was called with the SAME object
    if mock_dm.CommandSystem.execute_command.called:
        args, _ = mock_dm.CommandSystem.execute_command.call_args
        # signature: state, cmd, source, player, ctx
        passed_cmd = args[1]
        if passed_cmd is cmd_obj:
            print("SUCCESS: CommandSystem received the CommandDef object directly.")
        else:
            print(f"FAILURE: CommandSystem received {type(passed_cmd)} (expected original object).")
            sys.exit(1)
    else:
        print("FAILURE: CommandSystem.execute_command was not called.")
        sys.exit(1)

if __name__ == "__main__":
    test_direct_execution()
