import sys
sys.path.insert(0, 'bin\\Release')
import dm_ai_module as dm

card_db = dm.JsonLoader.load_cards('data/cards.json')
# pick first spell
spell_cards = [cid for cid, cdef in card_db.items() if getattr(cdef, 'type', None) == dm.CardType.SPELL]
if not spell_cards:
    print('no spells')
    raise SystemExit(1)
spell_id = spell_cards[0]
print('Using spell', spell_id)

game = dm.GameInstance(99, card_db)
config = dm.ScenarioConfig()
config.my_hand_cards = [spell_id]
config.my_mana_zone = [spell_id] * 5
config.my_shields = []
config.enemy_shield_count = 5

try:
    game.reset_with_scenario(config)
except Exception as e:
    print('reset_with_scenario raised', e)

# Ensure MAIN
try:
    game.state.current_phase = dm.Phase.MAIN
except Exception:
    pass

print('Before: pending_effects attr exists?', hasattr(game.state, 'pending_effects'))
print('Before: pending_effects len (if any):', len(getattr(game.state, 'pending_effects', [])))

actions = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
print('Generated actions count:', len(actions))
for i, a in enumerate(actions[:20]):
    print(i, type(a), vars(a) if hasattr(a, '__dict__') else a)

# Attempt to play first PLAY/DECLARE_PLAY-like action
play_actions = [a for a in actions if getattr(a, 'type', None) in (getattr(dm.ActionType, 'PLAY_CARD', None), getattr(dm.ActionType, 'DECLARE_PLAY', None))]
print('play_actions count:', len(play_actions))
if play_actions:
    try:
        game.resolve_action(play_actions[0])
        print('resolve_action called')
    except Exception as e:
        print('resolve_action raised', e)

print('After: pending_effects len:', len(getattr(game.state, 'pending_effects', [])))

actions2 = dm.IntentGenerator.generate_legal_actions(game.state, card_db)
print('After generated actions count:', len(actions2))
for i,a in enumerate(actions2[:20]):
    print('  ', i, type(a), getattr(a, 'type', None), getattr(a, 'source_instance_id', None), getattr(a, 'card_id', None))
