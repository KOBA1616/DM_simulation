import dm_ai_module

card_id = 9999
# create effect

try:
    eff = dm_ai_module.EffectDef()
    eff.trigger = dm_ai_module.TriggerType.ON_PLAY
    ad = dm_ai_module.ActionDef()
    ad.type = dm_ai_module.EffectActionType.DRAW_CARD
    eff.actions = [ad]
    # create card data
    cdata = dm_ai_module.CardData(card_id, 'Test', 3, 'WATER', 2000, 'CREATURE', ['Cyber Lord'], [eff])
    dm_ai_module.register_card_data(cdata)
    print('registered', dm_ai_module._CARD_REGISTRY.get(card_id) is not None)
except Exception as e:
    print('register failed', e)

# create game instance

gi = dm_ai_module.GameInstance(1)
state = gi.state
print('players:', len(state.players))
state.add_card_to_hand(0, card_id, 0)
print('hand before:', [(c.card_id,c.instance_id) for c in state.players[0].hand])
cmd = dm_ai_module.TransitionCommand(0, dm_ai_module.Zone.HAND, dm_ai_module.Zone.BATTLE, 0, -1)
cmd.execute(state)
print('hand after:', [(c.card_id,c.instance_id) for c in state.players[0].hand])
print('battle after:', [(c.card_id,c.instance_id) for c in state.players[0].battle])
print('pending:', state.get_pending_effects_info())
