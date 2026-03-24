#!/usr/bin/env python3
"""Parse TransitionCommand snapshot dumps and produce a summary JSON.

Usage:
  python scripts/parse_transition_snapshots.py --input-dir logs/transition_snapshots --output logs/transition_snapshots/summary.json

The snapshot format (written by TransitionCommand) is key:value lines, for example:
  card_instance_id:123
  owner_id:0
  from_zone:2
  to_zone:3
  owner_counts: battle=1,hand=4,mana=0,deck=35,grave=0
  source_entries:12:34,56:78,90:12

This script will parse those files and emit a machine-readable summary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List, Any


KV_RE = re.compile(r"^([a-zA-Z0-9_]+)\s*:\s*(.*)$")


def parse_owner_counts(s: str) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for part in s.split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            try:
                d[k.strip()] = int(v.strip())
            except Exception:
                d[k.strip()] = 0
    return d


def parse_source_entries(s: str) -> List[Dict[str, int]]:
    out: List[Dict[str, int]] = []
    items = [p.strip() for p in s.split(',') if p.strip()]
    for it in items:
        # format instance:card or instance:card:... (take first two)
        parts = it.split(':')
        try:
            inst = int(parts[0])
            card = int(parts[1]) if len(parts) > 1 else -1
        except Exception:
            continue
        out.append({"instance_id": inst, "card_id": card})
    return out


def parse_file(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {"file": path}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = KV_RE.match(line)
                if not m:
                    # fallback: skip
                    continue
                key, val = m.group(1), m.group(2)
                if key == 'owner_counts':
                    data['owner_counts'] = parse_owner_counts(val)
                elif key == 'source_entries':
                    data['source_entries'] = parse_source_entries(val)
                else:
                    # try int
                    try:
                        data[key] = int(val)
                    except Exception:
                        data[key] = val
    except Exception as e:
        data['__error'] = str(e)
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', default='logs/transition_snapshots')
    parser.add_argument('--output', '-o', default='logs/transition_snapshots/summary.json')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    files = glob(os.path.join(args.input_dir, '*.log'))

    parsed: List[Dict[str, Any]] = []
    for p in sorted(files):
        parsed.append(parse_file(p))

    # Build aggregate stats
    stats: Dict[str, Any] = {
        'file_count': len(parsed),
        'missing_instance_ids': {},
        'by_owner': {},
    }
    for item in parsed:
        cid = item.get('card_instance_id')
        owner = item.get('owner_id')
        if cid is not None:
            stats['missing_instance_ids'][str(cid)] = stats['missing_instance_ids'].get(str(cid), 0) + 1
        if owner is not None:
            stats['by_owner'][str(owner)] = stats['by_owner'].get(str(owner), 0) + 1

    out = {'files': parsed, 'stats': stats}
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f'Parsed {len(parsed)} files, wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
