import dm_ai_module
s = dm_ai_module.GameState()
s.players = [dm_ai_module.PlayerStub(), dm_ai_module.PlayerStub()]
s.players[0].hand = [dm_ai_module.CardStub(42, 1001)]
s.active_player_id = 0
outs = dm_ai_module.ActionGenerator.generate_legal_actions(s, None)
print('generated', len(outs))
for a in outs:
    print('action type', getattr(a,'type',None), 'card', getattr(a,'card_id',None), 'cmd', getattr(a,'command',None))
