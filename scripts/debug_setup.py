import sys, pathlib, os
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.environ['DM_DISABLE_NATIVE'] = '1'
import dm_ai_module as dm

print('Using dm_ai_module from', getattr(dm,'__file__',None))

db = dm.JsonLoader.load_cards(str(repo_root / 'data' / 'cards.json'))
print('Loaded cards:', len(db))

game = dm.GameInstance(0, db)
s = game.state
s.set_deck(0, [1]*40)
s.set_deck(1, [1]*40)

dm.PhaseManager.start_game(s, db)
print('Initial phase:', s.current_phase, 'active:', s.active_player_id)

for i in range(30):
    print(f'Iter {i}: phase={s.current_phase} active={s.active_player_id}')
    legal = dm.IntentGenerator.generate_legal_commands(s, db)
    print('  legal cmds:', [getattr(c,'type',None) for c in legal])
    if legal:
        game.resolve_command(legal[0])
    else:
        dm.PhaseManager.next_phase(s, db)
    if 'MAIN' in str(s.current_phase).upper() and s.active_player_id == 0:
        print('Reached MAIN')
        break
print('Final phase:', s.current_phase, 'active:', s.active_player_id)
