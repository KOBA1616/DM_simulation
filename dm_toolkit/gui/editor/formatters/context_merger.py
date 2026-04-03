from typing import Dict, Any, List, Callable

# --- Common helper predicates for pattern matching ---

def is_draw_item(it: Dict[str, Any]) -> bool:
    if not isinstance(it, dict): return False
    t = it.get('type', '')
    if t == 'DRAW_CARD': return True
    if t == 'TRANSITION':
        from_z = (it.get('from_zone') or it.get('fromZone') or '').upper()
        to_z = (it.get('to_zone') or it.get('toZone') or '').upper()
        if (from_z == '' or 'DECK' in from_z) and 'HAND' in to_z:
            return True
    return False

def is_deck_bottom_move(it: Dict[str, Any]) -> bool:
    if not isinstance(it, dict): return False
    dest = (it.get('destination_zone') or it.get('to_zone') or it.get('toZone') or '').upper()
    if 'DECK_BOTTOM' in dest or 'DECKBOTTOM' in dest:
        return True
    t = (it.get('type') or '').upper()
    if 'DECK_BOTTOM' in t: return True
    return False

def is_cast_spell_item(it: Dict[str, Any]) -> bool:
    if not isinstance(it, dict): return False
    return it.get('type', '').upper() == 'CAST_SPELL'

def is_replace_card_move(it: Dict[str, Any]) -> bool:
    if not isinstance(it, dict): return False
    return it.get('type', '').upper() == 'REPLACE_CARD_MOVE'

# --- Rule definitions ---

class ContextMerger:
    """Rule-based engine for merging consecutive actions into natural Japanese sentences."""

    # Declarative rules definition based on JSON-like structures
    RULES: List[Dict[str, Any]] = [
        {
            "match": ["CAST_SPELL", "REPLACE_CARD_MOVE"],
            "template": "その呪文を唱えた後、{from_zone}に置くかわりに{to_zone}に置く。"
        },
        {
            "match": ["DRAW_CARD", "DECK_BOTTOM"],  # Simplified custom match approach
            "template": "{0}その後、引いた枚数と同じ枚数を山札の下に置く。"
        }
    ]

    @classmethod
    def _matches_rule(cls, rule_match: List[str], items: List[Dict[str, Any]]) -> bool:
        if len(items) < len(rule_match):
            return False

        for i, match_type in enumerate(rule_match):
            if match_type == "CAST_SPELL" and not is_cast_spell_item(items[i]): return False
            if match_type == "REPLACE_CARD_MOVE" and not is_replace_card_move(items[i]): return False
            if match_type == "DRAW_CARD" and not is_draw_item(items[i]): return False
            if match_type == "DECK_BOTTOM" and not is_deck_bottom_move(items[i]): return False

        return True

    @classmethod
    def _apply_rule(cls, rule: Dict[str, Any], items: List[Dict[str, Any]], texts: List[str]) -> str:
        from dm_toolkit.gui.i18n import tr
        match_len = len(rule["match"])
        template = rule["template"]

        # Build kwargs for dynamic interpolation
        kwargs = {}
        for i, item in enumerate(items[:match_len]):
            # Expose properties using prefix indicating their step index, e.g. "0_amount"
            for k, v in item.items():
                if isinstance(v, str):
                    kwargs[f"{i}_{k}"] = tr(v)
                else:
                    kwargs[f"{i}_{k}"] = v

        # Backward compatibility aliases for common variables to make templates cleaner
        if "CAST_SPELL" in rule["match"] and "REPLACE_CARD_MOVE" in rule["match"]:
            kwargs["from_zone"] = tr(items[1].get('from_zone', 'GRAVEYARD'))
            kwargs["to_zone"] = tr(items[1].get('to_zone', 'DECK_BOTTOM'))

        # Prepare formatted text array interpolation (e.g. {0})
        # Use rstrip('。') + '。' to normalize sentence ends if needed
        format_args = [t.rstrip('。') + '。' for t in texts[:match_len]]

        merged = template.format(*format_args, **kwargs)

        if len(texts) > match_len:
            rest = ' '.join(texts[match_len:]).strip()
            if rest:
                merged = merged.rstrip('。') + '、' + rest
        return merged

    @classmethod
    def merge(cls, raw_items: List[Dict[str, Any]], formatted_texts: List[str]) -> str:
        """
        Iterate over registered rules. If a rule's condition matches the sequence,
        apply its merge function. Defaults to simply joining with spaces if no rules match.
        """
        if not formatted_texts:
            return ""

        for rule in cls.RULES:
            if cls._matches_rule(rule["match"], raw_items):
                return cls._apply_rule(rule, raw_items, formatted_texts)

        # Dynamic conjunction logic based on input/output dependencies
        valid_texts = [t for t in formatted_texts if t.strip()]
        if not valid_texts:
            return ""

        merged = valid_texts[0].rstrip('。') + "。"
        # find the actual index of the valid texts in the original lists to map raw_items correctly
        valid_indices = [i for i, t in enumerate(formatted_texts) if t.strip()]

        for i in range(1, len(valid_indices)):
            curr_idx = valid_indices[i]
            prev_idx = valid_indices[i-1]

            prev_item = raw_items[prev_idx]
            curr_item = raw_items[curr_idx]

            prev_out = prev_item.get("output_value_key")
            curr_in = curr_item.get("input_value_key") or curr_item.get("input_link")

            if prev_out and curr_in and prev_out == curr_in:
                merged += "その後、" + formatted_texts[curr_idx].lstrip('。')
            else:
                merged += "また、" + formatted_texts[curr_idx].lstrip('。')
            merged = merged.rstrip('。') + "。"

        return merged