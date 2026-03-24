from typing import Dict, Any, List, Callable

class ContextMerger:
    """Rule-based engine for merging consecutive actions into natural Japanese sentences."""

    RULES: List[Dict[str, Any]] = []

    @classmethod
    def register_rule(cls, condition_fn: Callable[[List[Dict[str, Any]]], bool],
                      merge_fn: Callable[[List[Dict[str, Any]], List[str]], str]):
        """Register a pattern matching rule."""
        cls.RULES.append({
            "condition": condition_fn,
            "merge": merge_fn
        })

    @classmethod
    def merge(cls, raw_items: List[Dict[str, Any]], formatted_texts: List[str]) -> str:
        """
        Iterate over registered rules. If a rule's condition matches the sequence,
        apply its merge function. Defaults to simply joining with spaces if no rules match.
        """
        if not formatted_texts:
            return ""

        for rule in cls.RULES:
            if rule["condition"](raw_items):
                return rule["merge"](raw_items, formatted_texts)

        return " ".join([t for t in formatted_texts if t]).strip()

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

def cond_spell_then_replace(items: List[Dict[str, Any]]) -> bool:
    return len(items) >= 2 and is_cast_spell_item(items[0]) and is_replace_card_move(items[1])

def merge_spell_then_replace(items: List[Dict[str, Any]], texts: List[str]) -> str:
    from dm_toolkit.gui.i18n import tr
    from_zone_key = items[1].get('from_zone', 'GRAVEYARD')
    to_zone_key = items[1].get('to_zone', 'DECK_BOTTOM')
    from_zone_text = tr(from_zone_key)
    to_zone_text = tr(to_zone_key)

    merged = f"その呪文を唱えた後、{from_zone_text}に置くかわりに{to_zone_text}に置く。"
    if len(texts) > 2:
        rest = ' '.join(texts[2:]).strip()
        if rest:
            merged = merged.rstrip('。') + '、' + rest
    return merged

ContextMerger.register_rule(cond_spell_then_replace, merge_spell_then_replace)


def cond_draw_then_bottom(items: List[Dict[str, Any]]) -> bool:
    return len(items) >= 2 and is_draw_item(items[0]) and is_deck_bottom_move(items[1])

def merge_draw_then_bottom(items: List[Dict[str, Any]], texts: List[str]) -> str:
    first = texts[0].rstrip('。')
    tail = 'その後、引いた枚数と同じ枚数を山札の下に置く。'
    merged = f"{first}。{tail}"
    if len(texts) > 2:
        rest = ' '.join(texts[2:]).strip()
        if rest:
            merged = merged.rstrip('。') + '、' + rest
    return merged

ContextMerger.register_rule(cond_draw_then_bottom, merge_draw_then_bottom)
