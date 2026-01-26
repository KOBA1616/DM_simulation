import random
import time
from dm_toolkit.gui.headless import create_session, run_steps
from dm_toolkit.commands import generate_legal_commands

LOG = 'tmp_selfplay_smoke.log'

def run_one_game(max_steps=200):
    sess = create_session()
    gs = sess.gs
    card_db = sess.card_db
    steps = 0
    exceptions = 0
    actions_executed = 0
    while steps < max_steps and not sess.is_game_over():
        try:
            cmds = generate_legal_commands(gs, card_db)
            if not cmds:
                # advance a phase to avoid stall
                try:
                    sess.step_phase()
                except Exception:
                    pass
                steps += 1
                continue
            # pick a non-pass preferably
            choice = None
            for c in cmds:
                try:
                    d = c.to_dict()
                except Exception:
                    d = {}
                t = d.get('type', '')
                if t and t != 'PASS':
                    choice = c
                    break
            if choice is None:
                choice = random.choice(cmds)
            try:
                sess.execute_action(choice)
                actions_executed += 1
            except Exception:
                exceptions += 1
        except Exception:
            exceptions += 1
        # advance phase/step
        try:
            sess.step_phase()
        except Exception:
            pass
        steps += 1
    return {
        'steps': steps,
        'exceptions': exceptions,
        'actions': actions_executed,
        'game_over': sess.is_game_over(),
        'winner': getattr(sess.gs, 'winner', None) if sess.gs else None,
    }


def main(games=10):
    random.seed(42)
    results = []
    start = time.time()
    for i in range(games):
        r = run_one_game()
        results.append(r)
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write(f'Game {i}: {r}\n')
    duration = time.time() - start
    summary = {
        'games': games,
        'duration_s': duration,
        'results': results,
    }
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('SUMMARY:\n')
        f.write(str(summary) + '\n')
    print('Done', summary)

if __name__ == '__main__':
    main(10)
