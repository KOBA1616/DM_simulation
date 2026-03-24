import json
from tools.cards_audit import audit_cost_modifier_fields_from_json


def test_cards_json_has_no_stat_scaled_missing_fields():
    """Scan data/cards.json and ensure no COST_MODIFIER/COST_REDUCTION has STAT_SCALED missing fields."""
    with open('data/cards.json', 'r', encoding='utf-8') as f:
        content = f.read()

    issues = audit_cost_modifier_fields_from_json(content)
    assert issues == [], f"Found STAT_SCALED field issues in data/cards.json: {issues}"
import json
from pathlib import Path


def load_cards():
    p = Path(__file__).resolve().parent.parent / 'data' / 'cards.json'
    assert p.exists(), f"cards.json not found at {p}"
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)


def test_stat_scaled_entries_have_required_fields():
    cards = load_cards()
    missing = []
    # cards.json expected to be a dict mapping card_id -> card_def or a list
    items = cards.items() if isinstance(cards, dict) else enumerate(cards)
    for key, card in items:
        try:
            statics = card.get('static_abilities') or []
        except Exception:
            continue
        for idx, s in enumerate(statics):
            if not isinstance(s, dict):
                continue
            if s.get('type') != 'COST_MODIFIER':
                continue
            if s.get('value_mode', 'FIXED') == 'STAT_SCALED':
                # require stat_key and per_value
                if 'stat_key' not in s or not isinstance(s.get('stat_key'), str) or not s.get('stat_key'):
                    missing.append((key, idx, 'stat_key'))
                if 'per_value' not in s or not isinstance(s.get('per_value'), int) or s.get('per_value') <= 0:
                    missing.append((key, idx, 'per_value'))

    if missing:
        msgs = [f"card={m[0]} static_idx={m[1]} missing={m[2]}" for m in missing]
        raise AssertionError("STAT_SCALED missing required fields:\n" + "\n".join(msgs))