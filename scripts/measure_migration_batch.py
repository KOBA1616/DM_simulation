"""Run measure_migration_legacy-style sampling across multiple seeds and strata,
aggregate results and write JSON/CSV summary.

Usage:
  python scripts/measure_migration_batch.py --input data/cards.json --count 100 \
      --strata era --seeds 1,42,100 --out reports/batch_era_100.json
"""
from __future__ import annotations
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random

# reuse analyzer from measure_migration_legacy
import sys
from pathlib import Path as _P
# Ensure scripts package path available when run as script
_p = _P(__file__).resolve().parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from measure_migration_legacy import analyze_card


def sample_counts(data: list, count: int, strata: str | None, seed: int) -> Counter:
    if seed is not None:
        random.seed(seed)

    if strata:
        groups: dict = {}
        for card in data:
            key = card.get(strata)
            groups.setdefault(key, []).append(card)

        total_cards = len(data)
        picked = []
        for key, group in groups.items():
            target = max(1, round(count * (len(group) / total_cards)))
            if len(group) <= target:
                picked.extend(group)
            else:
                picked.extend(random.sample(group, target))
        if len(picked) > count:
            picked = random.sample(picked, count)
    else:
        if count >= len(data):
            picked = list(data)
        else:
            try:
                picked = random.sample(list(data), count)
            except ValueError:
                picked = list(data)[:count]

    total = Counter()
    for card in picked:
        total.update(analyze_card(card))
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='data/cards.json')
    p.add_argument('--count', '-n', type=int, default=100)
    p.add_argument('--strata', '-s', default=None)
    p.add_argument('--seeds', default='42')
    p.add_argument('--out', '-o', default=None)
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input file not found: {path}")
        return 2

    data = json.loads(path.read_text(encoding='utf-8'))
    seeds = [int(x.strip()) for x in args.seeds.split(',') if x.strip()]

    per_seed = {}
    aggregate = Counter()
    per_token_values = defaultdict(list)

    for s in seeds:
        cnt = sample_counts(data, args.count, args.strata, s)
        per_seed[s] = dict(cnt)
        for k, v in cnt.items():
            per_token_values[k].append(v)
        aggregate.update(cnt)

    # Compute averages per token across seeds
    avg_per_token = {k: sum(vs) / len(vs) for k, vs in per_token_values.items()}

    result = {
        'input': str(path),
        'count': args.count,
        'strata': args.strata,
        'seeds': seeds,
        'per_seed_counts': per_seed,
        'aggregate_counts': aggregate.most_common(),
        'avg_per_token': avg_per_token,
    }

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Wrote batch results to {outp}")

        # also write CSV summary
        csvp = outp.with_suffix('.csv')
        with csvp.open('w', encoding='utf-8') as f:
            f.write('token,aggregate,avg\n')
            for k, v in result['aggregate_counts']:
                avg = result['avg_per_token'].get(k, 0)
                f.write(f'{k},{v},{avg}\n')
        print(f"Wrote batch CSV to {csvp}")

    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
