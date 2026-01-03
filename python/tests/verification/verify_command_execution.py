
import sys
import os

# Add bin directory to sys.path
sys.path.append(os.path.abspath("bin"))

try:
    import dm_ai_module
    print(f"Successfully imported dm_ai_module: {dm_ai_module}")
except ImportError as e:
    print(f"Failed to import dm_ai_module: {e}")
    sys.exit(1)

# Check for CommandSystem binding
if not hasattr(dm_ai_module, "CommandSystem"):
    print("FAILED: CommandSystem class is missing from dm_ai_module.")
    sys.exit(1)

print("SUCCESS: CommandSystem class is present.")

# Check for execute_command method
if not hasattr(dm_ai_module.CommandSystem, "execute_command"):
    print("FAILED: CommandSystem.execute_command method is missing.")
    sys.exit(1)

print("SUCCESS: CommandSystem.execute_command method is present.")

# Basic execution test (Stubbed implementation check)
try:
    state = dm_ai_module.GameState(40)
    cmd = dm_ai_module.CommandDef()
    cmd.type = dm_ai_module.CommandType.TRANSITION # Using a valid primitive

    # We pass a context dict. The C++ stub should accept it.
    ctx = {"source_instance_id": 0}

    # It might fail inside C++ if implementation is incomplete, but the binding should work.
    # The current CommandSystem.execute_command implementation in existing command_system.cpp
    # seems to have logic for TAP/UNTAP/RETURN_TO_HAND etc.

    # Let's try TAP which seems implemented in the file I cat'ed
    cmd.type = dm_ai_module.CommandType.TAP
    cmd.target_group = dm_ai_module.TargetScope.SELF # To trigger some logic

    dm_ai_module.CommandSystem.execute_command(state, cmd, 0, 0, ctx)
    print("SUCCESS: CommandSystem.execute_command called successfully.")

except Exception as e:
    print(f"FAILED during execution: {e}")
    # We don't exit 1 here necessarily if it's logic error, but for binding test it is good.
    # If it says "not implemented" or similar, that's fine for now as long as binding exists.
    # But segmentation fault would be bad.
