import random
import time
from dm_toolkit.gui.headless import create_session
from dm_toolkit.commands import generate_legal_commands

LOG = 'tmp_selfplay_long.log'

def run_one_game(max_steps=10000):
    sess = create_session()
    gs = sess.gs
    card_db = sess.card_db
    steps = 0
    exceptions = 0
    actions_executed = 0
    # counts for command types we care about
    cmd_counts = {
        'MANA_CHARGE': 0,
        'PLAY_FROM_ZONE': 0,
        'ATTACK': 0,
    }
    cmd_samples = {k: [] for k in cmd_counts}
    while steps < max_steps and not sess.is_game_over():
        try:
            cmds = generate_legal_commands(gs, card_db)
            if not cmds:
                try:
                    sess.step_phase()
                except Exception:
                    pass
                steps += 1
                continue
            # inspect normalized commands (if available) and count types
            for c in cmds:
                try:
                    d = c.to_dict()
                except Exception:
                    d = {}
                t = d.get('type') or getattr(c, 'type', None) or getattr(getattr(c, 'command', {}), 'get', lambda *_: '')('type', '')
                if t in cmd_counts:
                    cmd_counts[t] += 1
                    if len(cmd_samples[t]) < 5:
                        cmd_samples[t].append(d or repr(c))
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
        'cmd_counts': cmd_counts,
        'cmd_samples': cmd_samples,
    }


def main(games=1, max_steps=10000):
    random.seed(42)
    results = []
    start = time.time()
    for i in range(games):
        r = run_one_game(max_steps=max_steps)
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
    # run 1 long game by default
    main(games=1, max_steps=10000)
