import dm_ai_module, json
card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
print('6 in card_db:', 6 in card_db)
print('card_db[6] type:', type(card_db.get(6)))
print('card_db[6] effects:', getattr(card_db.get(6), 'effects', None))
if getattr(card_db.get(6), 'effects', None):
	for e in card_db[6].effects:
		print(' effect trigger=', getattr(e,'trigger',None), 'actions=', [(getattr(a,'type',None), getattr(a,'scope',None)) for a in getattr(e,'actions',[])])
gs = dm_ai_module.GameState(2)
gs.setup_test_duel()
gs.set_deck(0, [6,6,6])
dm_ai_module.DevTools.move_cards(gs,0,dm_ai_module.Zone.DECK,dm_ai_module.Zone.MANA,2,6)
dm_ai_module.DevTools.move_cards(gs,0,dm_ai_module.Zone.DECK,dm_ai_module.Zone.HAND,1,6)
gs.add_test_card_to_battle(1,1,5001,False,False)
act = dm_ai_module.Action(); act.type = dm_ai_module.ActionType.PLAY_CARD; act.card_id = 6
act.source_instance_id = gs.players[0].hand[0].instance_id
print('before declare hand:', [getattr(c,'instance_id',None) for c in gs.players[0].hand])
dm_ai_module.EffectResolver.resolve_action(gs, act, card_db)
print('after declare stack_zone len:', len(gs.stack_zone), 'stack0 attrs:', vars(gs.stack_zone[-1]) if gs.stack_zone else None)
pay = dm_ai_module.Action(); pay.type = dm_ai_module.ActionType.PAY_COST
stack_card = gs.stack_zone[-1]
pay.source_instance_id = stack_card.instance_id
print('pay.source_instance_id set to', pay.source_instance_id)
dm_ai_module.EffectResolver.resolve_action(gs, pay, card_db)
print('after pay top.paid=', getattr(gs.stack_zone[-1], 'paid', None) if gs.stack_zone else None)
res = dm_ai_module.Action(); res.type = dm_ai_module.ActionType.RESOLVE_PLAY
res.source_instance_id = stack_card.instance_id
dm_ai_module.EffectResolver.resolve_action(gs, res, card_db)
print('after resolve stack len:', len(gs.stack_zone))
print('_pending_effects:', gs.get_pending_effects_info())
print('players[1] battle ids:', [getattr(c,'instance_id',None) for c in gs.players[1].battle])
print('players[1] hand ids:', [getattr(c,'instance_id',None) for c in gs.players[1].hand])
