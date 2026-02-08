"""Test to verify _ActionWrapper has _action attribute"""
import sys
sys.path.insert(0, '.')

import dm_ai_module
from dm_toolkit import commands_v2

# Prefer the v2 command-first wrapper
generate_legal_commands = commands_v2.generate_legal_commands

# Setup game state
gs = dm_ai_module.GameState(42)
gs.setup_test_duel()
deck = [1,2,3,4,5,6,7,8,9,10]*4
gs.set_deck(0, deck)
gs.set_deck(1, deck)

# Load card database
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')

# Start game and progress to MANA phase
dm_ai_module.PhaseManager.start_game(gs, card_db)
dm_ai_module.PhaseManager.fast_forward(gs, card_db)

print(f"Current phase: {gs.current_phase}")

# Generate commands
cmds = generate_legal_commands(gs, card_db)
print(f"Generated {len(cmds)} commands")

if cmds:
    cmd = cmds[0]
    print(f"\nFirst command type: {type(cmd)}")
    print(f"Has _action attribute: {hasattr(cmd, '_action')}")
    
    if hasattr(cmd, '_action'):
        action = cmd._action
        print(f"_action type: {type(action)}")
        print(f"_action value: {action}")
        print(f"_action.type: {action.type if action else 'None'}")
    else:
        print("ERROR: Command does not have _action attribute!")
        print(f"Command attributes: {dir(cmd)}")
        
    # Also check to_dict()
    try:
        cmd_dict = cmd.to_dict()
        print(f"\nto_dict() result: {cmd_dict}")
    except Exception as e:
        print(f"to_dict() error: {e}")
