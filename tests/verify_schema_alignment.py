
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

# Verify new enums are present
expected_enums = [
    "ATTACK_PLAYER",
    "ATTACK_CREATURE",
    "RESOLVE_BATTLE",
    "MEKRAID",
    "LOOK_AND_ADD"
]

missing = []
for e in expected_enums:
    if not hasattr(dm_ai_module.CommandType, e):
        missing.append(e)

if missing:
    print(f"FAILED: Missing CommandType enums: {missing}")
    sys.exit(1)

print("SUCCESS: All new CommandType enums are present.")

# Verify CommandDef structure (basic)
try:
    cmd = dm_ai_module.CommandDef()
    cmd.type = dm_ai_module.CommandType.ATTACK_PLAYER
    print(f"Successfully created CommandDef with type {cmd.type}")
except Exception as e:
    print(f"Failed to create CommandDef: {e}")
    sys.exit(1)
