import sys, os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'bin'))
from dm_toolkit.engine.compat import EngineCompat
import dm_ai_module

state = dm_ai_module.GameState(40)
card_data = dm_ai_module.CardData(1, 'TestCard', 1, 'FIRE', 1000, 'CREATURE', ['Human'], [])
dm_ai_module.register_card_data(card_data)

p1 = 0
inst_id = 100
state.add_test_card_to_battle(p1, 1, inst_id, False, True)

print('Before: battle ids', [c.instance_id for c in state.players[p1].battle_zone])
print('Before: hand ids', [c.instance_id for c in state.players[p1].hand])
print('CommandSystem exists:', hasattr(dm_ai_module, 'CommandSystem'))
if hasattr(dm_ai_module,'CommandSystem'):
	print('CommandSystem.execute_command exists:', hasattr(dm_ai_module.CommandSystem, 'execute_command'))
print('CommandType members sample:', list(dm_ai_module.CommandType.__members__.keys())[:20])

# Build dict command for RETURN_TO_HAND
cmd = {'type': 'RETURN_TO_HAND', 'target_filter': {'zones': ['BATTLE_ZONE']}, 'target_group': 'SELF'}
print('\nCalling EngineCompat.ExecuteCommand with cmd dict...')
EngineCompat.ExecuteCommand(state, cmd)

print('After wrapper: battle ids', [c.instance_id for c in state.players[p1].battle_zone])
print('After wrapper: hand ids', [c.instance_id for c in state.players[p1].hand])
