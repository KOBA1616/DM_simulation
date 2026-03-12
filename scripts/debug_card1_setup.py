from pathlib import Path
import os
import sys
# ensure project root on sys.path for direct python runs
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dm_ai_module as dm
ROOT = Path(__file__).resolve().parents[1]
CARDS = str(ROOT / 'data' / 'cards.json')

def run():
    db = None
    try:
        db = dm.JsonLoader.load_cards(CARDS)
    except Exception as e:
        print('JsonLoader.load_cards failed:', e)
        db = {}
    game = dm.GameInstance(42, db)
    s = game.state
    s.set_deck(0, [1]*40)
    s.set_deck(1, [1]*40)
    dm.PhaseManager.start_game(s, db)
    # add mana
    for i in range(2):
        s.add_card_to_mana(0, 1, 9200 + i)
    # advance to MAIN
    for i in range(30):
        print(f'loop {i}: phase={s.current_phase} active={s.active_player_id} hand0={len(s.players[0].hand)}')
        if 'MAIN' in str(s.current_phase).upper() and s.active_player_id == 0:
            break
        legal = dm.IntentGenerator.generate_legal_commands(s, db)
        print('  legal cmds:', [getattr(c,'type',None) for c in legal])
        if legal:
            game.resolve_command(legal[0])
        else:
            dm.PhaseManager.next_phase(s, db)
    print('after setup: phase=', s.current_phase, 'active=', s.active_player_id)
    legal = dm.IntentGenerator.generate_legal_commands(s, db)
    print('final legal cmds:', [getattr(c,'type',None) for c in legal])

if __name__=='__main__':
    run()
