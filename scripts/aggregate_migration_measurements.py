"""Run multiple sampling runs of measure_migration_legacy logic across seeds/strata
and aggregate token counts.

Usage:
    python scripts/aggregate_migration_measurements.py -i data/cards.json -n 100

This script depends on `scripts/measure_migration_legacy.py`'s `analyze_card` logic.
"""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import random
from collections import Counter

# Import analyze_card from the other script
import importlib.util
from pathlib import Path as _P

# Import analyze_card from scripts/measure_migration_legacy.py regardless of module path
spec_path = _P(__file__).resolve().parent / 'measure_migration_legacy.py'
spec = importlib.util.spec_from_file_location('measure_migration_legacy', str(spec_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
analyze_card = mod.analyze_card
LEGACY_KEYS = getattr(mod, 'LEGACY_KEYS', None)


def sample_and_count(data, count, strata_key, seed):
    if seed is not None:
        random.seed(seed)

    # prepare cards_to_analyze similar to measure_migration_legacy
    if strata_key:
        groups = {}
        for card in data:
            key = card.get(strata_key)
            groups.setdefault(key, []).append(card)
        total_cards = len(data)
        cards_to_analyze = []
        for key, group in groups.items():
            target = max(1, round(count * (len(group) / total_cards)))
            if len(group) <= target:
                picked = list(group)
            else:
                picked = random.sample(group, target)
            cards_to_analyze.extend(picked)
        if len(cards_to_analyze) > count:
            cards_to_analyze = random.sample(cards_to_analyze, count)
    else:
        if count >= len(data):
            cards_to_analyze = list(data)
        else:
            try:
                cards_to_analyze = random.sample(list(data), count)
            except ValueError:
                cards_to_analyze = list(data)[:count]

    total = Counter()
    visited = 0
    for card in cards_to_analyze:
        total.update(analyze_card(card))
        visited += 1
    return visited, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='data/cards.json')
    p.add_argument('--count', '-n', type=int, default=100)
    p.add_argument('--seeds', default='42,7,99', help='Comma-separated seeds')
    p.add_argument('--strata', default='era,author,None', help='Comma-separated strata keys; use None for no strata')
    p.add_argument('--output', '-o', default=None)
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input file not found: {path}")
        return 2

    data = json.loads(path.read_text(encoding='utf-8'))
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    strata_raw = [s.strip() for s in args.strata.split(',') if s.strip()]
    strata = [None if s.lower() == 'none' else s for s in strata_raw]

    results = {}
    aggregate = Counter()

    for strata_key in strata:
        for seed in seeds:
            visited, counts = sample_and_count(data, args.count, strata_key, seed)
            key = f"strata={str(strata_key)};seed={seed}"
            results[key] = {
                'strata': strata_key,
                'seed': seed,
                'analyzed': visited,
                'counts': dict(counts)
            }
            aggregate.update(counts)

    summary = {
        'input': str(path),
        'runs': len(results),
        'total_analyzed': sum(v['analyzed'] for v in results.values()),
        'aggregate_counts': dict(aggregate)
    }

    out = {
        'summary': summary,
        'details': results
    }

    outp = Path(args.output) if args.output else Path('reports') / 'migration_aggregate.json'
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Wrote aggregate report to {outp}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
