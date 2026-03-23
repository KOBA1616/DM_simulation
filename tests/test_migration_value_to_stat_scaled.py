import json
from tools.cards_migration import migrate_value_to_stat_scaled


def test_migrate_value_to_stat_scaled_basic():
    card = {
        "id": 9999,
        "name": "Legacy Test",
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value": 2,
                "scope": "ALL"
            }
        ]
    }
    content = json.dumps([card], ensure_ascii=False)
    migrated = migrate_value_to_stat_scaled(content)
    parsed = json.loads(migrated)
    assert isinstance(parsed, list)
    mcard = parsed[0]
    statics = mcard.get('static_abilities') or []
    assert len(statics) == 1
    s = statics[0]
    assert s.get('value_mode') == 'STAT_SCALED'
    assert s.get('stat_key') == 'legacy_value'
    assert s.get('per_value') == 2
    assert s.get('min_stat') == 1
    assert s.get('max_reduction') == 2
    assert 'value' not in s
