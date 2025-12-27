import dm_ai_module

card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
print('card_db.get(6)=', card_db.get(6))
func = dm_ai_module.get_pending_effects_info
print('module get_pending_effects_info func:', func, 'name=', getattr(func, '__name__', None))
import inspect
try:
    print('SOURCE:\n', inspect.getsource(func))
except Exception as e:
    print('Could not get source:', e)

gs = dm_ai_module.GameState(2)
gs.setup_test_duel()

gs.set_deck(0, [6,6,6])
dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.MANA, 2, 6)
dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.HAND, 1, 6)

gs.add_test_card_to_battle(1, 1, 5001, False, False)

# Play
act = dm_ai_module.Action(); act.type = dm_ai_module.ActionType.PLAY_CARD; act.card_id = 6
act.source_instance_id = gs.players[0].hand[0].instance_id
print('Before PLAY, stack:', gs.stack_zone)
dm_ai_module.EffectResolver.resolve_action(gs, act, card_db)
print('After PLAY, stack:', gs.stack_zone)

# Pay
pay = dm_ai_module.Action(); pay.type = dm_ai_module.ActionType.PAY_COST
stack_card = gs.stack_zone[-1]
pay.source_instance_id = stack_card.instance_id
print('Before PAY, top.paid:', getattr(gs.stack_zone[-1], 'paid', None))
dm_ai_module.EffectResolver.resolve_action(gs, pay, card_db)
print('After PAY, top.paid:', getattr(gs.stack_zone[-1], 'paid', None))

# Resolve
res = dm_ai_module.Action(); res.type = dm_ai_module.ActionType.RESOLVE_PLAY
res.source_instance_id = stack_card.instance_id
print('Resolving play...')
dm_ai_module.EffectResolver.resolve_action(gs, res, card_db)

raw = gs._pending_effects
helper = dm_ai_module.get_pending_effects_info(gs)
print('Pending effects raw repr:')
for i,p in enumerate(raw):
    print('  slot', i, 'type=', type(p), '->', repr(p))
print('Pending effects via helper repr:')
for i,p in enumerate(helper):
    print('  slot', i, 'type=', type(p), '->', repr(p))
if helper:
    ptype = helper[0][0]
    print('ptype=', ptype)
    if ptype != 0:
        r = dm_ai_module.Action(); r.type = dm_ai_module.ActionType.RESOLVE_EFFECT; r.slot_index = 0
        dm_ai_module.EffectResolver.resolve_action(gs, r, card_db)
        helper = dm_ai_module.get_pending_effects_info(gs)
    if helper:
        sel = dm_ai_module.Action(); sel.type = dm_ai_module.ActionType.SELECT_TARGET; sel.slot_index = 0
        sel.target_instance_id = 5001
        dm_ai_module.EffectResolver.resolve_action(gs, sel, card_db)
        fin = dm_ai_module.Action(); fin.type = dm_ai_module.ActionType.RESOLVE_EFFECT; fin.slot_index = 0
        dm_ai_module.EffectResolver.resolve_action(gs, fin, card_db)

print('Players[1] hand:', gs.players[1].hand)
print('Players[1] battle:', gs.players[1].battle)
