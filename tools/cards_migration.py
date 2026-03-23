import json
from typing import Any, Dict


def migrate_value_to_stat_scaled(json_content: str) -> str:
    """Convert legacy COST_MODIFIER entries that use `value` into STAT_SCALED entries.

    Rules (minimal, conservative):
    - For each card, inspect `static_abilities` and `cost_reductions` lists.
    - If an entry has `type == 'COST_MODIFIER'` and contains an integer `value`
      and either no `value_mode` or `value_mode == 'FIXED'`, convert it to
      `value_mode = 'STAT_SCALED'` with `stat_key = 'legacy_value'`,
      `per_value = <value>`, `min_stat = 1`, and `max_reduction = <value>`.
    - Remove the old `value` field after migration.

    Returns the migrated JSON string (pretty-printed).
    """
    j = json.loads(json_content)

    def process_mod_list(mods: list):
        for m in mods:
            if not isinstance(m, dict):
                continue
            if m.get('type') != 'COST_MODIFIER':
                continue
            # legacy single `value` handling
            if 'value' in m and (m.get('value_mode') in (None, 'FIXED') ):
                try:
                    v = int(m.get('value', 0))
                except Exception:
                    v = 0
                # apply conservative conversion only for positive integers
                if v > 0:
                    m['value_mode'] = 'STAT_SCALED'
                    m['stat_key'] = 'legacy_value'
                    m['per_value'] = v
                    m['min_stat'] = 1
                    m['max_reduction'] = v
                    # remove legacy field to avoid ambiguity
                    del m['value']

    if isinstance(j, list):
        for item in j:
            if not isinstance(item, dict):
                continue
            static_abilities = item.get('static_abilities') or []
            cost_reductions = item.get('cost_reductions') or []
            process_mod_list(static_abilities)
            process_mod_list(cost_reductions)
    elif isinstance(j, dict):
        static_abilities = j.get('static_abilities') or []
        cost_reductions = j.get('cost_reductions') or []
        process_mod_list(static_abilities)
        process_mod_list(cost_reductions)

    return json.dumps(j, ensure_ascii=False, indent=2)


def migrate_file_inplace(path: str) -> None:
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    migrated = migrate_value_to_stat_scaled(content)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(migrated)
