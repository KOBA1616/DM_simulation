# -*- coding: utf-8 -*-
"""
Validate card editor generated texts and command templates for unsupported types
and poor localization candidates.

- Checks data/editor_templates.json entries against supported command/flow types.
- Reports entries that use editor-only or unknown types.
- Suggests localization gaps (e.g., missing translations for zones/types).

Run:
    python tools/validate_generated_texts.py
"""
import json
import os
from typing import Dict, Any, List, Set

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_PATH = os.path.join(WORKSPACE, 'data', 'editor_templates.json')

# Supported command types at schema level (without native module)
SUPPORTED_COMMAND_TYPES: Set[str] = {
    'TRANSITION', 'MUTATE', 'FLOW', 'QUERY', 'DECIDE', 'DECLARE_REACTION', 'STAT', 'GAME_RESULT',
    # Common legacy-ish that editors may proxy
    'DRAW_CARD', 'ADD_KEYWORD', 'POWER_MOD', 'MANA_CHARGE', 'SELECT_NUMBER', 'CHOICE',
}

SUPPORTED_FLOW_TYPES: Set[str] = {
    # Known core flows
    'PHASE_CHANGE', 'TURN_CHANGE', 'SET_ACTIVE_PLAYER',
    # Text generator references
    'STEP_CHANGE',
}

KNOWN_ZONES: Set[str] = {
    'BATTLE', 'MANA', 'SHIELD', 'DECK', 'HAND', 'GRAVEYARD', 'BUFFER', 'UNDER_CARD',
    'BATTLE_ZONE', 'MANA_ZONE', 'SHIELD_ZONE', 'DECK_TOP', 'DECK_BOTTOM'
}

ISSUES: List[str] = []


def load_templates() -> Dict[str, Any]:
    with open(TEMPLATES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_template_item(idx: int, item: Dict[str, Any]):
    name = item.get('name', f'Item#{idx}')
    data = item.get('data', {})
    ctype = str(data.get('type', 'NONE')).upper()
    editor_only = bool(data.get('editor_only', False))

    if ctype not in SUPPORTED_COMMAND_TYPES:
        if not editor_only:
            ISSUES.append(f"Unsupported command type in template: '{name}' -> type='{ctype}' (not marked editor_only)")
        else:
            ISSUES.append(f"Editor-only template: '{name}' -> type='{ctype}' (ok) ")
    # Flow-specific checks
    if ctype == 'FLOW':
        ftype = str(data.get('flow_type', '')).upper()
        if ftype and ftype not in SUPPORTED_FLOW_TYPES:
            if not editor_only:
                ISSUES.append(f"Unknown flow_type in template: '{name}' -> flow_type='{ftype}' (unsupported)")
            else:
                ISSUES.append(f"Editor-only flow_type in template: '{name}' -> flow_type='{ftype}' (ok)")

    # Zone checks
    for zkey in ('from_zone', 'to_zone', 'destination_zone'):
        z = data.get(zkey)
        if z and str(z).upper() not in KNOWN_ZONES:
            ISSUES.append(f"Unknown zone name in template: '{name}' -> {zkey}='{z}'")


def main():
    try:
        doc = load_templates()
    except Exception as e:
        print(f"Failed to load templates: {e}")
        return 1

    cmds = doc.get('commands', [])
    for i, it in enumerate(cmds):
        if isinstance(it, dict):
            check_template_item(i, it)
        else:
            ISSUES.append(f"Non-dict command template at index {i}")

    if not ISSUES:
        print('No issues found. Templates look consistent.')
        return 0

    print('Detected issues:')
    for msg in ISSUES:
        print(' - ' + msg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
