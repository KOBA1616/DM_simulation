import dm_ai_module
s = dm_ai_module.GameState()
s.players = [dm_ai_module.PlayerStub(), dm_ai_module.PlayerStub()]
s.players[0].hand = [dm_ai_module.CardStub(42, 1001)]
s.active_player_id = 0
from dm_toolkit.commands import generate_legal_commands
outs = generate_legal_commands(s, None)
print('generated', len(outs))
for a in outs:
    try:
        print('action dict', a.to_dict())
    except Exception:
        print('action repr', repr(a))
