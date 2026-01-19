#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run multiple games using `full_self_play.play_one_full` and collect statistics.
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from full_self_play import play_one_full


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--trace-first', action='store_true')
    args = parser.parse_args()

    counts = {}
    seed_base = args.seed if args.seed is not None else 0

    for i in range(args.games):
        seed = seed_base + i if args.seed is not None else None
        trace = args.trace_first and i == 0
        res = play_one_full(seed=seed, trace=trace)
        counts[res] = counts.get(res, 0) + 1
        if (i+1) % 50 == 0:
            print(f"Played {i+1}/{args.games}")

    out = {
        'counts': counts,
        'games': args.games,
    }
    out_dir = Path('reports')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'full_self_play_{args.games}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == '__main__':
    main()
