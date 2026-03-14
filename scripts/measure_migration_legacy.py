"""Measure migration branching by loading up to N legacy card JSON entries
and counting occurrences of legacy patterns that require conversion.

Usage:
    python scripts/measure_migration_legacy.py --input data/test_cards.json --count 10
"""
from __future__ import annotations
import json
import argparse
from collections import Counter
from pathlib import Path
import random

LEGACY_KEYS = [
    'ACTION', 'action', 'input_link', 'input_value_key', 'input_var',
    'output_link', 'output_value_key', 'output_var', 'flags'
]


def analyze_card(card: dict) -> Counter:
    c = Counter()

    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in LEGACY_KEYS:
                    c[k] += 1
                walk(v)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(card)
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='data/test_cards.json')
    p.add_argument('--count', '-n', type=int, default=10)
    p.add_argument('--strata', '-s', default=None,
                   help='Stratify sampling by card field, e.g. "era" or "author"')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed for reproducible sampling')
    p.add_argument('--output', '-o', default=None,
                   help='Optional output path (JSON) to write summary and sampled cards')
    p.add_argument('--csv', action='store_true', help='Also write a CSV summary of token counts')
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"Input file not found: {path}")
        return 2

    data = json.loads(path.read_text(encoding='utf-8'))
    total = Counter()

    # Prepare sampling list
    if args.seed is not None:
        random.seed(args.seed)

    cards_to_analyze = []
    if args.strata:
        # Group by strata key value (None used for missing)
        groups: dict = {}
        for card in data:
            key = card.get(args.strata)
            groups.setdefault(key, []).append(card)

        # Proportional allocation across strata
        total_cards = len(data)
        for key, group in groups.items():
            target = max(1, round(args.count * (len(group) / total_cards)))
            if len(group) <= target:
                picked = list(group)
            else:
                picked = random.sample(group, target)
            cards_to_analyze.extend(picked)

        # If rounding caused overshoot, trim randomly
        if len(cards_to_analyze) > args.count:
            cards_to_analyze = random.sample(cards_to_analyze, args.count)
    else:
        # Simple random or head sampling
        if args.count >= len(data):
            cards_to_analyze = list(data)
        else:
            # prefer random sample for larger datasets
            try:
                cards_to_analyze = random.sample(list(data), args.count)
            except ValueError:
                # fallback to head sampling
                cards_to_analyze = list(data)[: args.count]

    visited = 0
    for card in cards_to_analyze:
        cnt = analyze_card(card)
        total.update(cnt)
        visited += 1

    summary = {
        'analyzed': visited,
        'source': str(path),
        'counts': dict(total.most_common())
    }

    print(f"Analyzed {visited} cards from {path}")
    print("Legacy token counts:")
    for k, v in total.most_common():
        print(f"  {k}: {v}")

    if args.output:
        outp = Path(args.output)
        try:
            outp.parent.mkdir(parents=True, exist_ok=True)
            result = {'summary': summary, 'sampled_cards': cards_to_analyze}
            outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Wrote results to {outp}")
        except Exception as e:
            print(f"Failed to write output {outp}: {e}")
    if args.csv:
        # Write a simple CSV with token,count rows plus header summary
        try:
            csvp = Path('reports') / (Path(args.output).stem + '.csv' if args.output else 'migration_summary.csv')
            csvp.parent.mkdir(parents=True, exist_ok=True)
            with csvp.open('w', encoding='utf-8') as f:
                f.write('token,count\n')
                for k, v in total.most_common():
                    f.write(f'{k},{v}\n')
            print(f"Wrote CSV summary to {csvp}")
        except Exception as e:
            print(f"Failed to write CSV summary: {e}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())