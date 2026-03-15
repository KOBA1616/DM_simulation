import os
import sys

# Increase native logging verbosity
os.environ.setdefault('DM_CONSOLE_LOG_LEVEL', 'DEBUG')
os.environ.setdefault('DM_ROOT_LOG_LEVEL', 'DEBUG')

import sys
from pathlib import Path
# Ensure repo root on sys.path so dm_ai_module (package or pyd) is importable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import dm_ai_module
except Exception as e:
    print('Failed to import dm_ai_module:', e)
    raise

def repro():
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    cards_path = repo_root / 'data' / 'cards.json'
    print('Using cards.json:', cards_path)
    db = dm_ai_module.JsonLoader.load_cards(str(cards_path))

    CARD_ID = 13
    COST = 3

    game = dm_ai_module.GameInstance(42, db)
    s = game.state
    s.set_deck(0, [CARD_ID] * 40)
    s.set_deck(1, [1] * 40)
    dm_ai_module.PhaseManager.start_game(s, db)

    for i in range(COST):
        s.add_card_to_mana(0, CARD_ID, 9000 + i)

    # advance to MAIN P0
    for step in range(20):
        ph = str(s.current_phase).upper()
        print('Phase:', ph, 'active:', s.active_player_id)
        if 'MAIN' in ph and s.active_player_id == 0:
            print('Reached MAIN for P0')
            break
        legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        if legal:
            print('Generated legal command:', legal[0].type)
            game.resolve_command(legal[0])
        else:
            dm_ai_module.PhaseManager.next_phase(s, db)

    import json
    print('Pending before play:', s.get_pending_effect_count())
    legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
    play = None
    for c in legal:
        if 'PLAY' in str(getattr(c, 'type', '')).upper():
            play = c
            break

    print('Found play cmd:', bool(play))
    if play is None:
        print('No play command available')
        return

    print('Resolving play...')
    game.resolve_command(play)
    print('Pending after play:', s.get_pending_effect_count())

    # list pending effects content if any
    try:
        pe = s.get_pending_effects()
        print('Pending effects raw:', pe)
    except Exception as e:
        print('Could not list pending effects:', e)

if __name__ == '__main__':
    repro()
