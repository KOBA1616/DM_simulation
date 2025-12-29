import dm_ai_module as m
state = m.GameState(100)
# register card data
c1 = m.CardData(10, 'Fire Card', 1, 'FIRE', 1000, 'CREATURE', [], [])
try:
    m.register_card_data(c1)
except Exception:
    pass
state.add_card_to_mana(0,10,200)
state.add_card_to_mana(0,11,201)
state.add_card_to_mana(0,10,202)
state.add_card_to_mana(0,12,203)
# build effect
effect = m.EffectDef()
effect.trigger = m.TriggerType.ON_PLAY
act1 = m.ActionDef(); act1.type = m.EffectActionType.GET_GAME_STAT; act1.str_val = 'MANA_CIVILIZATION_COUNT'; act1.output_value_key='civ_count'
act2 = m.ActionDef(); act2.type = m.EffectActionType.DRAW_CARD; act2.input_value_key='civ_count'
act3 = m.ActionDef(); act3.type = m.EffectActionType.SEND_TO_DECK_BOTTOM; act3.scope = m.TargetScope.TARGET_SELECT; act3.filter = m.FilterDef(); act3.filter.zones=['HAND']; act3.input_value_key='civ_count'
effect.actions = [act1, act2, act3]
for i in range(10): state.add_card_to_deck(0,10,300+i)
print('before hand', len(state.players[0].hand))
ctx = m.GenericCardSystem.resolve_effect(state, effect, 999)
print('after hand', len(state.players[0].hand))
print('ctx', ctx)
