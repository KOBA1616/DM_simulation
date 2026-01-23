from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()/'bin'/'Release'))
import dm_ai_module as dm
card_db = dm.JsonLoader.load_cards('data/cards.json')
pr = dm.ParallelRunner(card_db, 10, 1)
print('runner ok')
evaler = dm.NeuralEvaluator(card_db)
print('evaluator ok')
gi = dm.GameInstance(0)
try:
    gi.state.setup_test_duel()
except Exception:
    pass
res = pr.play_games([gi.state], evaler, 1.0, False, 1, 0.0, False)
print('res type', type(res))
print('res len', len(res) if hasattr(res,'__len__') else 'no len')
