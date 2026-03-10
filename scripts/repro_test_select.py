import sys, pathlib, os
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
os.environ['DM_DISABLE_NATIVE'] = '1'
import dm_ai_module as dm

# Reproduce test_select_target_appears_after_draw (seed=10)

db = dm.JsonLoader.load_cards(str(repo_root / 'data' / 'cards.json'))

def _setup(card_id=1, mana_cost=2, seed=42):
    game = dm.GameInstance(seed, db)
    s = game.state
    s.set_deck(0, [card_id] * 40)
    s.set_deck(1, [1] * 40)
    dm.PhaseManager.start_game(s, db)
    for i in range(mana_cost):
        s.add_card_to_mana(0, card_id, 9200 + i)
    for _ in range(30):
        if 'MAIN' in str(s.current_phase).upper() and s.active_player_id == 0:
            break
        legal = dm.IntentGenerator.generate_legal_commands(s, db)
        print('loop: phase=', s.current_phase, 'legal=', [getattr(c,'type',None) for c in legal])
        if legal:
            game.resolve_command(legal[0])
        else:
            dm.PhaseManager.next_phase(s, db)
    return game, db


game, db = _setup(1,2,seed=10)
print('After setup: phase=', game.state.current_phase, 'active=', game.state.active_player_id)

# find PLAY
legal = dm.IntentGenerator.generate_legal_commands(game.state, db)
print('legal after setup:', [getattr(c,'type',None) for c in legal])
play_cmd = next((c for c in legal if 'PLAY' in str(getattr(c,'type','')).upper()), None)
print('play_cmd:', play_cmd)
if play_cmd is None:
    print('No PLAY command found, will exit')
else:
    print('Resolving PLAY')
    game.resolve_command(play_cmd)
    legal2 = dm.IntentGenerator.generate_legal_commands(game.state, db)
    print('after play legal:', [getattr(c,'type',None) for c in legal2])
