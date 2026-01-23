import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dm_toolkit.gui.game_session import GameSession

# simple callbacks
cb_update = lambda: None
cb_log = lambda m: print('LOG:', m)
cb_input = lambda: None
cb_action = lambda a: print('ACTION:', a)

sess = GameSession(cb_update, cb_log, cb_input, cb_action)
# Load card db via EngineCompat through GameWindow normally; provide empty dict
sess.card_db = {}

# Call reset_game (this uses dm_ai_module.GameState and our fallback)
print('Before reset:')
try:
    if sess.gs:
        print('Existing gs players:', len(sess.gs.players))
except Exception:
    pass

sess.reset_game()

if sess.gs:
    p0 = sess.gs.players[0]
    p1 = sess.gs.players[1]
    print('Deck sizes after reset: P0', len(p0.deck), 'P1', len(p1.deck))
    print('p0.deck type:', type(p0.deck))
    try:
        print('p0.deck dir sample:', dir(p0.deck)[:50])
    except Exception:
        pass
    print('P0 shields:', len(p0.shield_zone), 'hand:', len(p0.hand))
    print('P1 shields:', len(p1.shield_zone), 'hand:', len(p1.hand))
else:
    print('No GameState created')
