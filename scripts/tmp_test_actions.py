import dm_ai_module

gs = dm_ai_module.GameState()
print('initial phase', gs.current_phase)
# Put a card in hand and check main-phase actions
p = gs.players[0]
cs = dm_ai_module.CardStub(1, 101)
p.hand.append(cs)
gs.current_phase = dm_ai_module.Phase.MAIN
actions = dm_ai_module.ActionGenerator.generate_legal_actions(gs)
print('actions count', len(actions))
for a in actions:
    print('action', a.type, getattr(a,'card_id',None), getattr(a,'source_instance_id',None))

# Attack phase test
gs2 = dm_ai_module.GameState()
p2 = gs2.players[0]
c = dm_ai_module.CardStub(2,201)
c.is_tapped = False
c.sick = False
p2.battle_zone.append(c)
gs2.current_phase = dm_ai_module.Phase.ATTACK
acts = dm_ai_module.ActionGenerator.generate_legal_actions(gs2)
print('attack actions', len(acts))
for a in acts:
    print('attack action', a.type, getattr(a,'source_instance_id',None), getattr(a,'target_player',None))
