#!/usr/bin/env python3
"""Simple folder watcher that triggers the artifact manager when new files appear.
Uses polling to avoid extra dependencies so it works on Windows without watchdog.

Usage:
  python training/watch_and_trigger_artifact_manager.py --paths models data --interval 5 --timeout 300
"""
from __future__ import annotations
import time
import argparse
from pathlib import Path
import sys

# Import programmatic entry
import importlib.util
from pathlib import Path

# Load artifact_manager from file to avoid package import issues
_artifact_path = Path(__file__).resolve().parent / 'artifact_manager.py'
spec = importlib.util.spec_from_file_location('artifact_manager', str(_artifact_path))
artifact_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifact_manager)


def snapshot(paths):
    files = set()
    for p in paths:
        pp = Path(p)
        if pp.exists():
            for f in pp.rglob('*'):
                if f.is_file():
                    files.add(str(f.resolve()))
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+', default=['models', 'data'])
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    start = time.time()
    seen = snapshot(args.paths)
    print(f"Watching {args.paths} for new files (timeout {args.timeout}s) ...")
    try:
        while True:
            time.sleep(args.interval)
            current = snapshot(args.paths)
            new = current - seen
            if new:
                print(f"Detected {len(new)} new file(s). Triggering artifact manager...")
                try:
                    artifact_manager.run_artifact_manager(dry_run=args.dry_run)
                except Exception as e:
                    print("artifact manager failed:", e)
                seen = current
            if time.time() - start > args.timeout:
                print("Timeout reached, exiting watcher.")
                break
    except KeyboardInterrupt:
        print("Interrupted, exiting.")


if __name__ == '__main__':
    main()
