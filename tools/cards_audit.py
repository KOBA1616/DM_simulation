import json
from typing import List, Dict, Any


def audit_cost_modifier_fields_from_json(json_content: str) -> List[Dict[str, Any]]:
    """Audit cards JSON content for COST_MODIFIER / STAT_SCALED field issues.

    Returns list of issues: each item has keys: 'card_id', 'issue'
    """
    issues: List[Dict[str, Any]] = []
    try:
        j = json.loads(json_content)
    except Exception as e:
        issues.append({'card_id': None, 'issue': f'invalid_json: {e}'})
        return issues

    items = j if isinstance(j, list) else [j]
    for item in items:
        try:
            cid = item.get('id', None)
            # Check static_abilities and cost_reductions blocks
            static_abilities = item.get('static_abilities', []) or []
            cost_reductions = item.get('cost_reductions', []) or []

            # Helper inspect modifiers list-like
            def inspect_mods(mods, field_name):
                for idx, m in enumerate(mods):
                    # value_mode may be missing -> treat as FIXED
                    vm = m.get('value_mode', 'FIXED')
                    if vm == 'STAT_SCALED':
                        missing = []
                        if 'stat_key' not in m or not m.get('stat_key'):
                            missing.append('stat_key')
                        if 'per_value' not in m:
                            missing.append('per_value')
                        if missing:
                            issues.append({
                                'card_id': cid,
                                'issue': f"{field_name}[{idx}] value_mode=STAT_SCALED missing:{','.join(missing)}",
                            })

            inspect_mods(static_abilities, 'static_abilities')
            inspect_mods(cost_reductions, 'cost_reductions')

        except Exception as e:
            issues.append({'card_id': item.get('id', None) if isinstance(item, dict) else None, 'issue': f'audit_error: {e}'})

    return issues


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('file', help='cards.json file to audit')
    args = p.parse_args()
    with open(args.file, 'r', encoding='utf-8') as f:
        content = f.read()
    issues = audit_cost_modifier_fields_from_json(content)
    if not issues:
        print('No issues found')
        return 0
    for it in issues:
        print(f"card_id={it['card_id']} issue={it['issue']}")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
