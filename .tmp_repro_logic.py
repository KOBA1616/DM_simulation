import dm_ai_module as m
from dm_toolkit import dm_ai_module as mm

state = m.GameState(100)
native_state = getattr(state,'_native', state)
print('initial hand len', len(state.players[0].hand))
# Prepare db
db = m.CardRegistry.get_all_cards()
m.initialize_card_stats(native_state, db, 100)
# setup deck
native_state.add_card_to_deck(0,1,10)
native_state.add_card_to_deck(0,1,11)
native_state.add_card_to_deck(0,1,12)
native_state.add_card_to_deck(0,1,13)
# register card
card_data = m.CardData(1, 'Test Card', 1, m.Civilization.FIRE, 1000, m.CardType.CREATURE, [], [])
m.register_card_data(card_data)
# setup hand
native_state.add_card_to_hand(0,1,0)
print('after setup hand len', len(state.players[0].hand))
# create action IF_ELSE
cond_true = m.ConditionDef()
cond_true.type = 'COMPARE_STAT'
cond_true.stat_key = 'MY_HAND_COUNT'
cond_true.op = '>='
cond_true.value = 1

action = m.ActionDef()
action.type = m.EffectPrimitive.IF_ELSE
action.condition = cond_true
action.target_player = 'PLAYER_SELF'

then_act = m.ActionDef(); then_act.type = m.EffectPrimitive.DRAW_CARD; then_act.value1 = 1
else_act = m.ActionDef(); else_act.type = m.EffectPrimitive.DRAW_CARD; else_act.value1 = 2
action.options = [[then_act],[else_act]]

print('About to resolve_action')
m.GenericCardSystem.resolve_action(native_state, action, 0)
print('after resolve hand len', len(state.players[0].hand))
print('native hand len', len(native_state.players[0].hand))
print('native hand ids', [getattr(c,'instance_id',None) for c in native_state.players[0].hand])
print('proxy hand ids', [getattr(c,'instance_id',None) for c in state.players[0].hand])
