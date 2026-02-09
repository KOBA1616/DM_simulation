import sys
sys.path.insert(0, 'bin\\Release')
import dm_ai_module as dm

card_db = dm.JsonLoader.load_cards('data/cards.json')

# pick spell id 7 if exists
spell_cards = [cid for cid, cdef in card_db.items() if cdef.type == dm.CardType.SPELL]
if not spell_cards:
    print('no spells')
    sys.exit(1)
spell = spell_cards[0]
print('using spell', spell)

game = dm.GameInstance(99, card_db)
config = dm.ScenarioConfig()
config.my_hand_cards = [spell]
spell_def = card_db[spell]
config.my_mana_zone = [spell] * (spell_def.cost + 1)
config.my_shields = []
config.enemy_shield_count = 5

game.reset_with_scenario(config)
game.state.current_phase = dm.Phase.MAIN

print('before pending len', len(game.state.pending_effects))

from dm_toolkit import commands_v2 as commands
try:
    actions = commands.generate_legal_commands(game.state, card_db, strict=False) or []
except Exception:
    try:
        actions = commands.generate_legal_commands(game.state, card_db) or []
    except Exception:
        actions = []
print('initial actions count', len(actions))
for i,a in enumerate(actions[:20]):
    print(i, type(a), getattr(a, 'type', None), getattr(a, 'source_instance_id', None), getattr(a, 'slot_index', None))

# Play the spell
play = [a for a in actions if int(getattr(a, 'type', 0)) == 15]
if not play:
    print('no declare_play')
    sys.exit(0)
print('resolving declare_play')

game.resolve_action(play[0])

print('after play pending len', len(game.state.pending_effects))
print('pending raw:', game.state.pending_effects)
print('waiting_for_user_input:', getattr(game.state, 'waiting_for_user_input', None))
print('pending_query:', getattr(game.state, 'pending_query', None))
pq = game.state.pending_query
try:
    print('  query_type:', getattr(pq, 'query_type', None))
    print('  valid_targets:', getattr(pq, 'valid_targets', None))
    print('  params:', getattr(pq, 'params', None))
    print('  options:', getattr(pq, 'options', None))
except Exception as e:
    print('  failed to read pending_query fields', e)

# Now generate actions
actions2 = commands.generate_legal_commands(game.state, card_db, strict=False)
print('after-play actions count', len(actions2))
for i,a in enumerate(actions2[:50]):
    print(i, type(a), getattr(a, 'type', None), getattr(a, 'source_instance_id', None), getattr(a, 'slot_index', None))
