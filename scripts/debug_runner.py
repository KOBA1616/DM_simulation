#!/usr/bin/env python3
import os
import sys
import random

# Ensure we can import compiled extension from bin/ or project root
sys.path.insert(0, os.getcwd())
if os.path.isdir('python'):
    sys.path.insert(0, os.path.abspath('python'))

import dm_ai_module

card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
valid_ids = list(card_db.keys())
if not valid_ids:
    print('no cards')
    raise SystemExit(1)

deck = [random.choice(valid_ids) for _ in range(40)]
runner = dm_ai_module.ParallelRunner(card_db, 50, 1)

n = 200
res = runner.play_deck_matchup(list(deck), list(deck), n, 4)
print('C++ batch results:')
print('  total', len(res))
print('  count 1:', res.count(1))
print('  count 2:', res.count(2))
print('  count 0:', res.count(0))
print('  sample first 40:', res[:40])


def run_single_game_py(seed, deck_a, deck_b, max_steps=10000):
    inst = dm_ai_module.GameInstance(seed, card_db)
    gs = inst.state
    gs.set_deck(0, deck_a)
    gs.set_deck(1, deck_b)
    # Prefer using GameInstance.start_game to ensure pipeline/setup is consistent
    try:
        inst.start_game()
    except Exception:
        dm_ai_module.PhaseManager.start_game(gs, card_db)

    # Prefer native C++ HeuristicAgent if available
    try:
        Heur = dm_ai_module.HeuristicAgent
        ag0 = Heur(0, card_db)
        ag1 = Heur(1, card_db)
    except Exception:
        from dm_toolkit.ai.agent.heuristic_agent import HeuristicAgent
        ag0 = HeuristicAgent(0)
        ag1 = HeuristicAgent(1)

    steps = 0
    while steps < max_steps:
        if gs.winner != dm_ai_module.GameResult.NONE:
            break
        res = dm_ai_module.PhaseManager.check_game_over(gs, dm_ai_module.GameResult.DRAW)
        # check_game_over may set winner; re-evaluate
        if gs.winner != dm_ai_module.GameResult.NONE:
            break

        legal = dm_ai_module.IntentGenerator.generate_legal_actions(gs, card_db)
        if not legal:
            # Advance phases until actions appear or game ends (manual fast-forward)
            ff_steps = 0
            while True:
                if gs.winner != dm_ai_module.GameResult.NONE:
                    break
                legal2 = dm_ai_module.IntentGenerator.generate_legal_actions(gs, card_db)
                if legal2:
                    break
                dm_ai_module.PhaseManager.next_phase(gs, card_db)
                ff_steps += 1
                if ff_steps > 16:
                    break
            steps += 1
            continue

        if gs.active_player_id == 0:
            action = ag0.get_action(gs, legal)
        else:
            action = ag1.get_action(gs, legal)

        if action is None:
            dm_ai_module.PhaseManager.next_phase(gs, card_db)
            steps += 1
            continue

        try:
            inst.resolve_action(action)
        except Exception:
            try:
                dm_ai_module.GameLogicSystem.resolve_action_oneshot(gs, action, card_db)
            except Exception:
                pass

        steps += 1

    if gs.winner == dm_ai_module.GameResult.P1_WIN:
        return 1
    if gs.winner == dm_ai_module.GameResult.P2_WIN:
        return 2
    return 0


print('\nPython-driven single-game sampling (200 games)...')
counts = {0:0,1:0,2:0}
total_py = 200
n1 = total_py // 2
n2 = total_py - n1
for i in range(n1):
    seed = 2000 + i
    r = run_single_game_py(seed, list(deck), list(deck), max_steps=2000)
    counts[r] = counts.get(r,0) + 1
for i in range(n2):
    seed = 3000 + i
    # swap deck order so challenger is P2
    r = run_single_game_py(seed, list(deck), list(deck), max_steps=2000)
    # when swapped, a P2-win corresponds to challenger win; count as P1-equivalent by mapping
    counts[r] = counts.get(r,0) + 1

print('Py runner counts:', counts)
