import dm_ai_module as m

state = m.GameState(100)
native = getattr(state, '_native', state)
db = m.CardRegistry.get_all_cards()
m.initialize_card_stats(native, db, 100)

native.add_card_to_deck(0, 1, 10)
native.add_card_to_deck(0, 1, 11)
native.add_card_to_deck(0, 1, 12)
native.add_card_to_deck(0, 1, 13)

card_data = m.CardData(1, "Test", 1, m.Civilization.FIRE, 1000, m.CardType.CREATURE, [], [])
m.register_card_data(card_data)

native.add_card_to_hand(0, 1, 0)
print("before native hand:", list(getattr(native.players[0], 'hand', [])))
try:
    ph = state.players[0].hand
    try:
        print('before proxy hand type:', type(ph), 'len:', len(ph))
    except Exception:
        print('before proxy hand cannot len')
except Exception:
    print('no proxy')

action = m.ActionDef()
action.type = m.EffectPrimitive.IF_ELSE
cond = m.ConditionDef()
cond.type = "COMPARE_STAT"
cond.stat_key = "MY_HAND_COUNT"
cond.op = ">="
cond.value = 1
action.condition = cond
action.target_player = "PLAYER_SELF"
then_act = m.ActionDef(); then_act.type = m.EffectPrimitive.DRAW_CARD; then_act.value1 = 1
else_act = m.ActionDef(); else_act.type = m.EffectPrimitive.DRAW_CARD; else_act.value1 = 2
action.options = [[then_act], [else_act]]

print('calling resolve_action')
m.GenericCardSystem.resolve_action(native, action, 0)
print('after native hand:', list(getattr(native.players[0], 'hand', [])))
try:
    ph = state.players[0].hand
    try:
        print('after proxy hand type:', type(ph), 'len:', len(ph))
        for x in ph:
            print('proxy item type:', type(x))
    except Exception as e:
        print('after proxy hand cannot len, err', e)
except Exception:
    print('no proxy')
