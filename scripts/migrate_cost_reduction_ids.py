"""Migrate card JSON files to ensure each cost_reduction has an `id` field.

Usage:
    python scripts/migrate_cost_reduction_ids.py input.json output.json

If output omitted, writes to `<input>.migrated.json`.
"""
import sys
import json
from pathlib import Path


def ensure_ids(cards):
    """Ensure each dict in cards list has cost_reductions with ids.

    Non-dict entries are skipped (useful for deck lists of ints/strings).
    """
    for idx, card in enumerate(cards):
        if not isinstance(card, dict):
            # skip non-dict entries (e.g., deck lists of card ids)
            continue
        crs = card.get('cost_reductions') or []
        if not isinstance(crs, list):
            continue
        for i, cr in enumerate(crs):
            if not isinstance(cr, dict):
                continue
            if not cr.get('id'):
                cr['id'] = f"cr_{card.get('id','unknown')}_{i}"
            if not cr.get('name'):
                cr['name'] = cr['id']
    return cards


def main():
    if len(sys.argv) < 2:
        print('Usage: migrate_cost_reduction_ids.py input.json [output.json]')
        sys.exit(2)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else inp.with_suffix(inp.suffix + '.migrated')

    data = json.loads(inp.read_text(encoding='utf8'))
    # handle either list of cards or top-level object with 'cards' key
    if isinstance(data, dict):
        # common pattern: {'cards': [...]}
        if 'cards' in data and isinstance(data['cards'], list):
            data['cards'] = ensure_ids(data['cards'])
        else:
            # attempt to discover any top-level list of dicts and apply there
            modified = False
            for k, v in list(data.items()):
                if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                    data[k] = ensure_ids(v)
                    modified = True
            if not modified:
                raise SystemExit('Unrecognized JSON structure; no card list found')
    elif isinstance(data, list):
        data = ensure_ids(data)
    else:
        raise SystemExit('Unrecognized JSON structure; expected list or object with card lists')

    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')
    print('Wrote migrated JSON to', out)


if __name__ == '__main__':
    main()
