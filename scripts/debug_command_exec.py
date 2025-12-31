import sys, os
sys.path.append(os.path.join(os.getcwd(), 'bin'))
try:
    import dm_ai_module
except Exception as e:
    print('dm_ai_module import failed:', e)
    raise

state = dm_ai_module.GameState(40)
card_data = dm_ai_module.CardData(1, 'TestCard', 1, 'FIRE', 1000, 'CREATURE', ['Human'], [])
dm_ai_module.register_card_data(card_data)

p1 = 0
inst_id = 100
state.add_test_card_to_battle(p1, 1, inst_id, False, True)

print('Before: battle zone ids:', [c.instance_id for c in state.players[p1].battle_zone])
print('Before: hand ids:', [c.instance_id for c in state.players[p1].hand])

cmd_def = dm_ai_module.CommandDef()
cmd_def.type = dm_ai_module.CommandType.RETURN_TO_HAND
f = dm_ai_module.FilterDef()
f.zones = ['BATTLE_ZONE']
cmd_def.target_filter = f
cmd_def.target_group = dm_ai_module.TargetScope.SELF

print('Executing RETURN_TO_HAND via CommandSystem')
dm_ai_module.CommandSystem.execute_command(state, cmd_def, -1, p1, {})

print('After: battle zone ids:', [c.instance_id for c in state.players[p1].battle_zone])
print('After: hand ids:', [c.instance_id for c in state.players[p1].hand])

cmd_def2 = dm_ai_module.CommandDef()
cmd_def2.type = dm_ai_module.CommandType.TAP
f2 = dm_ai_module.FilterDef(); f2.zones = ['BATTLE_ZONE']; f2.owner = 'SELF'
cmd_def2.target_filter = f2
cmd_def2.target_group = dm_ai_module.TargetScope.SELF

print('Executing TAP via CommandSystem')
dm_ai_module.CommandSystem.execute_command(state, cmd_def2, -1, p1, {})
inst = state.get_card_instance(inst_id)
print('Instance after TAP: is_tapped =', inst.is_tapped)

print('Executing UNTAP via CommandSystem on same instance')
cmd_def3 = dm_ai_module.CommandDef(); cmd_def3.type = dm_ai_module.CommandType.UNTAP
f3 = dm_ai_module.FilterDef(); f3.zones=['BATTLE_ZONE']; f3.is_tapped=True
cmd_def3.target_filter = f3
cmd_def3.target_group = dm_ai_module.TargetScope.SELF

dm_ai_module.CommandSystem.execute_command(state, cmd_def3, -1, p1, {})
inst = state.get_card_instance(inst_id)
print('Instance after UNTAP: is_tapped =', inst.is_tapped)
