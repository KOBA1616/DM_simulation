import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../python')))
import dm_ai_module

with open('data/cards.json','r',encoding='utf-8') as f:
    dm_ai_module.card_registry_load_from_json(f.read())
card_db = dm_ai_module.CsvLoader.load_cards('data/cards.csv')
print('card_db[1].type =', card_db[1].type)
print('card_db[6].type =', card_db[6].type)

gs = dm_ai_module.GameState(2)
gs.setup_test_duel()

# Prepare deck and mana for Spiral Gate
gs.set_deck(0, [6,6,6])
dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.MANA, 2, 6)
dm_ai_module.DevTools.move_cards(gs, 0, dm_ai_module.Zone.DECK, dm_ai_module.Zone.HAND, 1, 6)

# Add a creature to opponent battle zone
gs.add_test_card_to_battle(1, 1, 5001, False, False)

print('Before play, pending:', dm_ai_module.get_pending_effects_info(gs))
print('Before play, pending verbose:', dm_ai_module.get_pending_effects_verbose(gs))

# Play spiral gate
act = dm_ai_module.Action(); act.type = dm_ai_module.ActionType.PLAY_CARD; act.card_id = 6
act.source_instance_id = gs.players[0].hand[0].instance_id

dm_ai_module.EffectResolver.resolve_action(gs, act, card_db)
print('After play, pending:', dm_ai_module.get_pending_effects_info(gs))
print('After play, pending verbose:', dm_ai_module.get_pending_effects_verbose(gs))

print('Player0 hand ids:', [c.instance_id for c in gs.players[0].hand])
print('Player0 grave ids:', [c.instance_id for c in gs.players[0].graveyard])
print('Player0 mana ids:', [c.instance_id for c in gs.players[0].mana_zone])
# Resolve pending effect (spell)
if dm_ai_module.get_pending_effects_info(gs):
    r = dm_ai_module.Action(); r.type = dm_ai_module.ActionType.RESOLVE_EFFECT; r.slot_index = 0
    dm_ai_module.EffectResolver.resolve_action(gs, r, card_db)
    print('After first resolve, pending:', dm_ai_module.get_pending_effects_info(gs))
    print('After first resolve, pending verbose:', dm_ai_module.get_pending_effects_verbose(gs))

# If a target selection pending remains, auto-select and resolve
if dm_ai_module.get_pending_effects_info(gs):
    sel = dm_ai_module.Action(); sel.type = dm_ai_module.ActionType.SELECT_TARGET; sel.slot_index = 0
    sel.target_instance_id = 5001
    dm_ai_module.EffectResolver.resolve_action(gs, sel, card_db)
    print('After select target, pending:', dm_ai_module.get_pending_effects_info(gs))
    print('After select target, pending verbose:', dm_ai_module.get_pending_effects_verbose(gs))
    fin = dm_ai_module.Action(); fin.type = dm_ai_module.ActionType.RESOLVE_EFFECT; fin.slot_index = 0
    dm_ai_module.EffectResolver.resolve_action(gs, fin, card_db)
    print('After final resolve, pending:', dm_ai_module.get_pending_effects_info(gs))
    print('After final resolve, pending verbose:', dm_ai_module.get_pending_effects_verbose(gs))

# Now confirm opponent creature moved to hand
opp = gs.players[1]
print('Opponent hand ids:', [c.instance_id for c in opp.hand])
print('Opponent battle ids:', [c.instance_id for c in opp.battle_zone])
print('Opponent grave ids:', [c.instance_id for c in opp.graveyard])
