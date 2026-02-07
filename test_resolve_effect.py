#!/usr/bin/env python3
"""Test RESOLVE_EFFECT command execution"""

import dm_ai_module
from dm_toolkit.action_to_command import map_action

# Create a RESOLVE_EFFECT action (simulating what C++ sends)
action = dm_ai_module.Action()
action.type = dm_ai_module.PlayerIntent.RESOLVE_EFFECT
action.slot_index = 0  # Effect index

print(f"Action created: type={action.type} (value={action.type.value}) name={action.type.name}")
print(f"  slot_index={action.slot_index}")

# Convert to command dict
cmd_dict = map_action(action)
print(f"\nCommand dict:")
print(f"  {cmd_dict}")

# Check if it's correctly typed
if cmd_dict.get('type') == 'RESOLVE_EFFECT':
    print("\n✓ SUCCESS: RESOLVE_EFFECT command correctly created!")
    print(f"  effect_index={cmd_dict.get('effect_index')}")
else:
    print(f"\n✗ FAIL: Expected type='RESOLVE_EFFECT', got type='{cmd_dict.get('type')}'")
