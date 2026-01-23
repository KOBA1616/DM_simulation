#!/usr/bin/env python3
"""Load ONNX model into native NeuralEvaluator and run ParallelRunner as a smoke test."""
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

def main(onnx_path: str = None):
    import dm_ai_module as dm
    from pathlib import Path

    models_dir = Path(repo_root) / 'models'
    if onnx_path is None:
        candidates = sorted(models_dir.glob('duel_transformer_*.onnx'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise SystemExit('No ONNX model found in models/')
        onnx_path = str(candidates[0])

    print('Loading ONNX model into NeuralEvaluator:', onnx_path)
    # Load card DB for evaluator constructor
    try:
        card_db = dm.JsonLoader.load_cards('data/cards.json')
    except Exception:
        card_db = {}
    evaluator = dm.NeuralEvaluator(card_db)
    try:
        evaluator.load_model(onnx_path)
    except Exception as e:
        print('NeuralEvaluator.load_model failed:', e)
        raise

    try:
        evaluator.set_model_type(dm.ModelType.TRANSFORMER)
    except Exception:
        pass

    # Create ParallelRunner (use small defaults) - prefer constructor with card_db
    pr = dm.ParallelRunner(card_db, 20, 1)

    # Prepare one initial GameState
    gi = dm.GameInstance(0)
    try:
        gi.state.setup_test_duel()
    except Exception:
        try:
            gi.start_game()
        except Exception:
            pass

    initial_states = [gi.state]

    print('Running ParallelRunner.play_games with evaluator (evaluation)')
    try:
        results = pr.play_games(initial_states, evaluator, 1.0, True, 2, 0.0, True)
    except Exception as e:
        print('ParallelRunner.play_games threw:', e)
        raise

    # Interpret results to compute a simple win_rate metric (P1 win fraction)
    total = 0
    p1_wins = 0
    try:
        if isinstance(results, list):
            for r in results:
                # GameResultInfo-like object
                if hasattr(r, 'result'):
                    val = r.result
                else:
                    val = r
                total += 1
                if val == dm.GameResult.P1_WIN:
                    p1_wins += 1
        else:
            # single result object
            if hasattr(results, 'result'):
                total = 1
                p1_wins = 1 if results.result == dm.GameResult.P1_WIN else 0
    except Exception as e:
        print('Failed to interpret play_games results:', e)

    win_rate = (p1_wins / total) if total > 0 else 0.0
    print(f'Evaluation: total_games={total}, p1_wins={p1_wins}, win_rate={win_rate:.3f}')

    # Save metadata next to ONNX
    try:
        meta = (Path(onnx_path).with_suffix('.json'))
        import json
        json.dump({'win_rate': win_rate}, open(meta, 'w'))
        print('Saved metadata:', meta)
    except Exception as e:
        print('Failed to save metadata:', e)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', type=str, default=None)
    args = p.parse_args()
    main(args.onnx)
