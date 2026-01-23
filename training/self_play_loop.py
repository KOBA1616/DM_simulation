#!/usr/bin/env python3
"""Self-play -> Generate -> Train loop

Usage:
  python training/self_play_loop.py --iterations 3 --episodes 12 --epochs 1
"""
import sys
import os
from pathlib import Path
import argparse
import time
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--episodes', type=int, default=12)
    parser.add_argument('--run-for-seconds', type=float, default=None, help='Stop the loop after this many seconds (overrides iterations if reached)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--parallel', type=int, default=8, help='Number of parallel games per head-to-head batch')
    parser.add_argument('--out-dir', type=str, default='data')
    parser.add_argument('--keep-data', type=int, default=5, help='Number of recent data files to keep')
    parser.add_argument('--keep-models', type=int, default=3, help='Number of recent model files to keep')
    parser.add_argument('--win-threshold', type=float, default=0.5, help='Win rate threshold under which models are considered low-performing')
    args = parser.parse_args()

    # Ensure built extension is preferred (if present)
    built_dir = repo_root / 'bin' / 'Release'
    if built_dir.exists():
        sys.path.insert(0, str(built_dir))

    try:
        import dm_ai_module as dm
    except Exception as e:
        print('Failed to import dm_ai_module:', e)
        raise

    # Import training function
    try:
        from training.train_simple import train_simple
    except Exception:
        # train_simple may be a script; import by path
        from importlib import import_module
        mod = import_module('training.train_simple')
        train_simple = getattr(mod, 'train_simple')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def prune_files(path_glob, keep):
        files = sorted(path_glob, key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[keep:]:
            try:
                old.unlink()
                print(f'    Removed old file: {old}')
            except Exception as e:
                print(f'    Failed to remove {old}:', e)

    start_time = time.time()

    for it in range(1, args.iterations + 1):
        # stop early if time limit reached
        if args.run_for_seconds is not None:
            elapsed = time.time() - start_time
            if elapsed >= args.run_for_seconds:
                print(f"Time limit reached (elapsed {elapsed:.1f}s >= {args.run_for_seconds}s). Stopping loop.")
                break

        print(f"\n== Iteration {it}/{args.iterations} \u000b collecting {args.episodes} episodes ==")
        collector = dm.DataCollector()
        batch = collector.collect_data_batch_heuristic(args.episodes, True, False)

        states = list(getattr(batch, 'token_states', []))
        policies = list(getattr(batch, 'policies', []))
        values = list(getattr(batch, 'values', []))

        if len(states) == 0:
            print('Warning: no samples collected this iteration')

        # Save to file
        out_path = out_dir / f'transformer_training_data_iter{it}.npz'
        states_np = np.array(states, dtype=np.int64)
        policies_np = np.array(policies, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32).reshape(-1, 1)

        np.savez_compressed(out_path, states=states_np, policies=policies_np, values=values_np)
        print(f'  Saved data: {out_path}  shapes: {states_np.shape}, {policies_np.shape}, {values_np.shape}')

        # Prune old data files
        try:
            data_glob = list(out_dir.glob('transformer_training_data_iter*.npz'))
            prune_files(sorted(data_glob), args.keep_data)
        except Exception as e:
            print('  Data pruning failed:', e)

        # Train on saved file
        print(f'== Iteration {it}: training {args.epochs} epoch(s) on generated data ==')
        train_simple(str(out_path), epochs=args.epochs, batch_size=args.batch_size)

        # Export latest trained model to ONNX and evaluate head-to-head vs current best
        try:
            from training.export_model_to_onnx import find_latest_model, export_to_onnx
            from training.head2head import head2head
            import shutil, json

            models_dir = repo_root / 'models'
            latest = find_latest_model(models_dir)
            if latest is not None:
                onnx_path = models_dir / (latest.stem + '.onnx')
                print('Exporting model to ONNX:', onnx_path)
                export_to_onnx(latest, onnx_path)
                print('ONNX export done, running head-to-head evaluation...')

                best_onnx = models_dir / 'best.onnx'
                promote = False
                if not best_onnx.exists():
                    # no baseline: promote challenger
                    shutil.copy2(str(onnx_path), str(best_onnx))
                    promote = True
                    win_rate = 1.0
                    wins = {1:1}
                else:
                    # run head-to-head: challenger (onnx_path) vs best (best_onnx)
                    games = max(20, args.episodes)
                    win_rate, wins = head2head(str(onnx_path), str(best_onnx), games=games, parallel=args.parallel)
                    print(f'  Challenger win rate vs best: {win_rate:.3f} (wins: {wins})')
                    if win_rate > args.win_threshold:
                        shutil.copy2(str(onnx_path), str(best_onnx))
                        promote = True

                # write metadata next to onnx (and keep metadata for pth with same stem)
                meta = {
                    'win_rate': float(win_rate),
                    'promoted': bool(promote),
                    'timestamp': time.time(),
                }
                try:
                    with open(str(onnx_path) + '.json', 'w') as f:
                        json.dump(meta, f)
                except Exception:
                    pass

                if promote:
                    # if promoted, also copy the trained pth as best.pth for bookkeeping
                    try:
                        best_pth = models_dir / 'best.pth'
                        shutil.copy2(str(latest), str(best_pth))
                    except Exception:
                        pass
        except Exception as e:
            print('Export/eval step failed:', e)

        # Prune old models saved by training (prefer removing low-win older models)
        try:
            models_dir = repo_root / 'models'
            if models_dir.exists():
                model_files = sorted(models_dir.glob('duel_transformer_*.pth'))

                # Build metadata map from corresponding .onnx.json if available
                low_win_models = []
                for p in model_files:
                    meta = models_dir / (p.stem + '.onnx.json')
                    # also check .onnx.json (export writes .onnx then .json)
                    meta_alt = models_dir / (p.stem + '.json')
                    win = None
                    try:
                        import json
                        if meta.exists():
                            d = json.load(open(meta))
                            win = float(d.get('win_rate', 0.0))
                        elif meta_alt.exists():
                            d = json.load(open(meta_alt))
                            win = float(d.get('win_rate', 0.0))
                    except Exception:
                        win = None
                    if win is not None and win < 0.5:
                        low_win_models.append(p)

                # If there are low-win models, remove oldest among them first until keep_models satisfied
                total_models = len(model_files)
                to_remove = []
                if total_models > args.keep_models and low_win_models:
                    # sort low-win by mtime ascending (oldest first)
                    low_sorted = sorted(low_win_models, key=lambda x: x.stat().st_mtime)
                    # remove as many as needed but not exceeding low_win_models count
                    remove_needed = total_models - args.keep_models
                    to_remove = low_sorted[:remove_needed]
                else:
                    # fallback: remove oldest models by age
                    if total_models > args.keep_models:
                        to_remove = model_files[:(total_models - args.keep_models)]

                for old in to_remove:
                    try:
                        old.unlink()
                        print(f'    Removed old model file: {old}')
                    except Exception as e:
                        print(f'    Failed to remove model {old}:', e)
        except Exception as e:
            print('  Model pruning failed:', e)

        # small pause to avoid tight loop
        time.sleep(1)

    print('\nSelf-play loop complete')


if __name__ == '__main__':
    main()
