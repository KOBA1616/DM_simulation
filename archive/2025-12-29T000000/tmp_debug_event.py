import dm_ai_module
from types import SimpleNamespace

print('CARD_REGISTRY keys before:', list(dm_ai_module._CARD_REGISTRY.keys())[:10])

# create effect and card data
card_id = 9999
effect_def = dm_ai_module.EffectDef()
effect_def.trigger = dm_ai_module.TriggerType.ON_PLAY
action_def = dm_ai_module.ActionDef()
action_def.type = dm_ai_module.EffectActionType.DRAW_CARD
effect_def.actions = [action_def]

cdata = dm_ai_module.CardData(
    card_id,
    "Test CIP Creature",
    3,
    "WATER",
    2000,
    "CREATURE",
    ["Cyber Lord"],
    [effect_def]
)

print('type(cdata) before register:', type(cdata), getattr(cdata, '__dict__', None))
dm_ai_module.register_card_data(cdata)
print('CARD_REGISTRY keys after:', list(dm_ai_module._CARD_REGISTRY.keys())[:10])
cdef_after = dm_ai_module._CARD_REGISTRY.get(9999)
print('type(cdef stored):', type(cdef_after), getattr(cdef_after, '__dict__', None))

# Create GameInstance and state
game = dm_ai_module.GameInstance(1)
state = game.state

player_id = 0
instance_id = 0
state.add_card_to_hand(player_id, card_id, instance_id)
print('hand after add:', [ (c.card_id, c.instance_id) for c in state.players[player_id].hand ])

cmd = dm_ai_module.TransitionCommand(instance_id, dm_ai_module.Zone.HAND, dm_ai_module.Zone.BATTLE, player_id, -1)
state.execute_command(cmd)
print('hand after transition:', [ (c.card_id, c.instance_id) for c in state.players[player_id].hand ])
print('battle after transition:', [ (c.card_id, c.instance_id) for c in state.players[player_id].battle ])
print('pending effects:', state.get_pending_effects_info())

# Inspect card definition from registry
cdef = dm_ai_module._CARD_REGISTRY.get(9999)
print('cdef:', cdef)
try:
    print('cdef.effects:', getattr(cdef, 'effects', None))
    for eff in getattr(cdef, 'effects', []) or []:
        print(' eff.trigger repr:', repr(getattr(eff, 'trigger', None)), ' type:', type(getattr(eff, 'trigger', None)))
except Exception as e:
    print('error inspecting cdef.effects', e)
