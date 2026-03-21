"""Migration helper: convert COST_MODIFIER entries from FIXED to STAT_SCALED form.

Rules:
- For entries with value_mode == 'FIXED' and a numeric 'value', convert to:
  - value_mode: 'STAT_SCALED'
  - stat_key: choose a reasonable default mapping (e.g., 'CREATURES_PLAYED') if condition absent
  - per_value: set to original 'value'
  - min_stat: 1
- Preserve condition/filter fields.
- If already STAT_SCALED, return unchanged.
"""
from typing import Dict, Any


def migrate_card(card: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(card, dict):
        return card
    out = dict(card)
    statics = out.get('static_abilities') or []
    new_statics = []
    for s in statics:
        if not isinstance(s, dict):
            new_statics.append(s)
            continue
        if s.get('type') == 'COST_MODIFIER':
            vm = s.get('value_mode', 'FIXED')
            if vm == 'FIXED' and 'value' in s:
                try:
                    val = int(s.get('value'))
                except Exception:
                    new_statics.append(s)
                    continue
                new = dict(s)
                new['value_mode'] = 'STAT_SCALED'
                # if condition/filter references a stat_key-like measure, reuse; else default
                # Simple heuristic: prefer CREATURES_PLAYED for battle-related filters
                cond = s.get('condition') or {}
                stat_key = None
                if isinstance(cond, dict):
                    f = cond.get('filter') or {}
                    races = f.get('races') or []
                    if races:
                        stat_key = 'CREATURES_PLAYED'
                if stat_key is None:
                    stat_key = 'GENERIC_USAGE'
                new['stat_key'] = stat_key
                new['per_value'] = val
                new['min_stat'] = 1
                # remove legacy 'value'
                if 'value' in new:
                    del new['value']
                new_statics.append(new)
            else:
                new_statics.append(s)
        else:
            new_statics.append(s)
    out['static_abilities'] = new_statics
    return out
