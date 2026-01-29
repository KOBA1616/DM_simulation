#!/usr/bin/env python3
"""Run self-play loop for a limited time and prune data/model files automatically.

Behavior:
- Launches `training/self_play_loop.py` with `--run-for-seconds` (default 300s).
- After run completes, prunes old data/model files by count and by total size limits.

Usage:
  python training/run_selfplay_with_prune.py
"""
import subprocess
import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF_PLAY = ROOT / 'training' / 'self_play_loop.py'

def human_size(n):
    for unit in ['B','KB','MB','GB']:
        if n < 1024.0:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"

def prune_by_count(path: Path, pattern: str, keep: int):
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    removed = []
    for old in files[keep:]:
        try:
            old.unlink()
            removed.append(old)
        except Exception:
            continue
    return removed

def prune_by_size(path: Path, pattern: str, max_bytes: int):
    files = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    removed = []
    while total > max_bytes and files:
        old = files.pop(0)
        try:
            sz = old.stat().st_size
            old.unlink()
            removed.append(old)
            total -= sz
        except Exception:
            continue
    return removed

def main():
    run_seconds = 300
    keep_data = 3
    keep_models = 2
    max_data_bytes = 200 * 1024 * 1024  # 200 MB
    max_models_bytes = 1024 * 1024 * 1024  # 1 GB

    cmd = [sys.executable, str(SELF_PLAY), '--run-for-seconds', str(run_seconds), '--episodes', '24', '--epochs', '1', '--batch-size', '8', '--parallel', '1', '--keep-data', str(keep_data), '--keep-models', str(keep_models)]
    print('Running self-play for', run_seconds, 'seconds...')
    start = time.time()
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # stream stdout
        while True:
            line = p.stdout.readline()
            if line == '' and p.poll() is not None:
                break
            if line:
                print(line, end='')
        rc = p.wait()
    except KeyboardInterrupt:
        p.terminate()
        rc = p.wait()

    elapsed = time.time() - start
    print(f"Self-play finished (rc={rc}) elapsed={elapsed:.1f}s")

    # Prune data and models by count and size
    data_dir = ROOT / 'data'
    models_dir = ROOT / 'models'
    removed = []
    if data_dir.exists():
        removed += prune_by_count(data_dir, 'transformer_training_data_iter*.npz', keep_data)
        removed += prune_by_size(data_dir, 'transformer_training_data_iter*.npz', max_data_bytes)
    if models_dir.exists():
        removed += prune_by_count(models_dir, 'duel_transformer_*.pth', keep_models)
        removed += prune_by_size(models_dir, 'duel_transformer_*.pth', max_models_bytes)

    if removed:
        print('Pruned files:')
        for r in removed:
            try:
                print(' ', r, human_size(r.stat().st_size))
            except Exception:
                print(' ', r)
    else:
        print('No files pruned')

if __name__ == '__main__':
    main()
