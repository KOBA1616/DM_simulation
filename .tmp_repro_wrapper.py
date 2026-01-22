import dm_ai_module as m
state = m.GameState(100)
native = getattr(state,'_native',state)
print('init wrapper hand len', len(state.players[0].hand))
print('init native hand len', len(native.players[0].hand))
native.add_card_to_deck(0,1,10)
native.add_card_to_deck(0,1,11)
native.add_card_to_mana(0,1,0)
action = m.ActionDef(); action.type = m.EffectPrimitive.IF; action.target_player = 'PLAYER_SELF'
filter_def = m.FilterDef(); filter_def.zones=['MANA_ZONE']; filter_def.civilizations = m.CivilizationList([m.Civilization.FIRE]); action.filter = filter_def
then_act = m.ActionDef(); then_act.type = m.EffectPrimitive.DRAW_CARD; then_act.value1=1
action.options = [[then_act]]
print('before resolve: wrapper._native is native?', getattr(state,'_native') is native)
print('before resolve: wrapper.players[0]._p is native.players[0]?', getattr(state.players[0],'_p') is native.players[0])
m.GenericCardSystem.resolve_action(native, action, 0)
print('after native hand len', len(native.players[0].hand))
print('after wrapper hand len', len(state.players[0].hand))
print('after resolve: wrapper.players[0]._p is native.players[0]?', getattr(state.players[0],'_p') is native.players[0])
