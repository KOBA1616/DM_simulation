import dm_ai_module as dm
from types import SimpleNamespace
# create effect and card data
ed = dm.EffectDef()
ed.trigger = dm.TriggerType.ON_PLAY
ad = dm.ActionDef()
ad.type = dm.EffectActionType.DRAW_CARD
ed.actions = [ad]
cd = dm.CardData(9999, 'Test CIP', 3, 'WATER', 2000, 'CREATURE', ['Cyber Lord'], [ed])
dm.register_card_data(cd)
print('CARD_REGISTRY keys:', list(dm._CARD_REGISTRY.keys()))
print('Registered item type:', type(dm._CARD_REGISTRY.get(9999)))
# setup game and state
gi = dm.GameInstance(1)
state = gi.state
state.add_card_to_hand(0, 9999, 0)
print('hand contains:', [(c.card_id, c.instance_id) for c in state.players[0].hand])
# execute transition
cmd = dm.TransitionCommand(0, dm.Zone.HAND, dm.Zone.BATTLE, 0, -1)
state.execute_command(cmd)
print('pending effects after transition:', state.get_pending_effects_info())
# inspect card definition stored
cdef = dm._CARD_REGISTRY.get(9999)
print('cdef effects:', getattr(cdef, 'effects', None))
for eff in getattr(cdef, 'effects', []) or []:
    print('eff.trigger type:', type(getattr(eff,'trigger',None)), 'value:', getattr(eff,'trigger',None))
