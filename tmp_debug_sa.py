import dm_ai_module
from dm_ai_module import GameState, CardDefinition

s = GameState(100)
# Ensure player exists
p = s.players[0]
# add sick creature
s.add_test_card_to_battle(0, 1, 10, False, True)
ci = s.players[0].battle[0]
print('ci attrs:', ci.__dict__)
# register cdef
cdef = dm_ai_module.CardDefinition(1, 'Vanilla', 'FIRE', [], ['Human'], 1, 1000, dm_ai_module.CardKeywords(), [])
card_db = {1: cdef}
# set phase and active player
s.active_player_id = 0
s.current_phase = dm_ai_module.Phase.ATTACK
acts = dm_ai_module.ActionGenerator.generate_legal_actions(s, card_db)
print('has attack actions:', any(a.type == dm_ai_module.ActionType.ATTACK_PLAYER for a in acts))
print('actions count', len(acts))
try:
    from dm_toolkit.commands_new import generate_legal_commands
except Exception:
    generate_legal_commands = None
if generate_legal_commands:
    cmds = generate_legal_commands(s, card_db)
    print('commands count', len(cmds))
else:
    cmds = []
for a in acts:
    print('action', a.type, getattr(a,'source_instance_id', None))
