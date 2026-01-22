import dm_ai_module as m
state = m.GameState(100)
native_state = getattr(state, '_native', state)
db = m.CardRegistry.get_all_cards()
m.initialize_card_stats(native_state, db, 100)
native_state.add_card_to_deck(0, 1, 10)
native_state.add_card_to_deck(0, 1, 11)
cd = m.CardData(1, 'Test Card', 1, m.Civilization.FIRE, 1000, m.CardType.CREATURE, [], [])
m.register_card_data(cd)
native_state.add_card_to_mana(0,1,0)
action = m.ActionDef(); action.type = m.EffectPrimitive.IF; action.target_player = 'PLAYER_SELF'
filter_def = m.FilterDef(); filter_def.zones=['MANA_ZONE']; filter_def.civilizations = m.CivilizationList([m.Civilization.FIRE]); action.filter = filter_def
then_act = m.ActionDef(); then_act.type = m.EffectPrimitive.DRAW_CARD; then_act.value1 = 1
action.options = [[then_act]]
print('before native hand len:', len(native_state.players[0].hand))
print('before wrapper hand len:', len(state.players[0].hand))
m.GenericCardSystem.resolve_action(native_state, action, 0)
print('after native hand len:', len(native_state.players[0].hand))
print('after wrapper hand len:', len(state.players[0].hand))
print('native hand repr:', native_state.players[0].hand)
print('wrapper hand repr:', state.players[0].hand)
