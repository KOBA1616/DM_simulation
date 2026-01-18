import os, sys, random
sys.path.insert(0, os.getcwd())
if os.path.isdir('python'):
    sys.path.insert(0, os.path.abspath('python'))
import dm_ai_module

card_db = dm_ai_module.JsonLoader.load_cards('data/cards.json')
valid_ids = list(card_db.keys())
if not valid_ids:
    print('no cards'); raise SystemExit(1)

deck = [random.choice(valid_ids) for _ in range(40)]
runner = dm_ai_module.ParallelRunner(card_db, 50, 1)

N = 200
res = runner.play_deck_matchup(list(deck), list(deck), N, 4)
print('C++ batch: ', {0:res.count(0), 1:res.count(1), 2:res.count(2)})

# For unresolved (0), run python single-game to try to complete
from debug_runner import run_single_game_py

needed = res.count(0)
print('Retrying', needed, 'games with Python runner')
counts = {0:res.count(0), 1:res.count(1), 2:res.count(2)}
for i in range(needed):
    seed = 10000 + i
    r = run_single_game_py(seed, deck, deck, max_steps=5000)
    counts[r] = counts.get(r,0) + 1
    counts[0] -= 1

print('Aggregated counts (after retries):', counts)
print('Win-rate (challenger):', (counts.get(1,0)+counts.get(2,0))/N)
