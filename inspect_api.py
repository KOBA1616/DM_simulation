#!/usr/bin/env python3
import dm_ai_module

# Check Phase enum
print("Phase enum values:")
for attr in dir(dm_ai_module.Phase):
    if not attr.startswith('_'):
        print(f"  - {attr}")

# Check GameCommand
print("\nGameCommand class:")
print("  Type:", type(dm_ai_module.GameCommand))

# Try to check how to create command
attrs = [a for a in dir(dm_ai_module.GameCommand) if not a.startswith('_')]
print("  Methods/Attributes:", attrs)

# Check get_pending_effects_info
print("\nGameState methods that might return effects:")
game = dm_ai_module.GameInstance(42)
gs = game.state
game.start_game()

# Try get_pending_effects_info
try:
    info = gs.get_pending_effects_info()
    print(f"  get_pending_effects_info: {type(info)}, len={len(info) if hasattr(info, '__len__') else 'N/A'}")
except Exception as e:
    print(f"  get_pending_effects_info error: {e}")
