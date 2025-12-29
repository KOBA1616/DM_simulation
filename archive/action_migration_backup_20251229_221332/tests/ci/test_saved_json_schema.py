import json
import os
from pathlib import Path

# Basic expectations per command type (minimal required keys)
REQUIRED_FIELDS = {
    "TRANSITION": ["to_zone"],
    "DESTROY": [],
    "DISCARD": [],
    "RETURN_TO_HAND": [],
    "MANA_CHARGE": [],
    "DRAW_CARD": ["amount"],
    "QUERY": ["str_param"],
    "MUTATE": ["mutation_kind"],
    "ADD_KEYWORD": ["mutation_kind"],
    "CHOICE": ["amount"],
    "SEARCH_DECK": ["amount"],
    "BREAK_SHIELD": ["amount"],
    "LOOK_AND_ADD": ["look_count"],
    "MEKRAID": ["look_count", "select_count"],
    "REGISTER_DELAYED_EFFECT": ["value1"],
    "PLAY_FROM_ZONE": ["to_zone"],
    "REVEAL_CARDS": ["amount"],
    "SHUFFLE_DECK": [],
    "ATTACH": ["base_target"],
    "RESET_INSTANCE": [],
    "SUMMON_TOKEN": ["amount", "token_id"],
    "FRIEND_BURST": ["str_val"],
    "CAST_SPELL": []
}

# Common alternate keys seen in legacy/converted files
KEY_ALIASES = {
    "amount": ["amount", "value1", "value2"],
    "to_zone": ["to_zone", "destination_zone"],
    "look_count": ["look_count", "value1", "amount"],
    "token_id": ["token_id", "str_val"],
    "str_param": ["str_param", "str_val"],
    "mutation_kind": ["mutation_kind", "str_val"],
}


def collect_card_files(data_dir="data"):
    p = Path(data_dir)
    if not p.exists():
        return []
    files = list(p.rglob("*.json"))
    return files


def iter_commands_in_card(card_json):
    # Card JSON may store command-like entries under various containers
    if not isinstance(card_json, dict):
        return

    container_keys = [
        "effects",
        "abilities",
        "triggers",
        "reaction_abilities",
        "static_abilities",
        "reaction_triggers",
    ]

    for key in container_keys:
        items = card_json.get(key, [])
        if not isinstance(items, list):
            continue
        for eff in items:
            if not isinstance(eff, dict):
                continue
            # effect may have actions or commands
            actions = eff.get("actions") or eff.get("commands") or []
            if not isinstance(actions, list):
                continue
            for a in actions:
                if isinstance(a, dict):
                    yield a
                    # Options: nested lists or dicts
                    opts = a.get("options") or a.get("choices") or []
                    if isinstance(opts, list):
                        for opt in opts:
                            if isinstance(opt, dict):
                                yield opt
                            elif isinstance(opt, list):
                                for sub in opt:
                                    if isinstance(sub, dict):
                                        yield sub
                else:
                    # Skip primitives or unexpected structures
                    continue


def test_saved_json_commands_have_required_fields():
    files = collect_card_files("data")
    assert files, "No card JSON files found under data/ — cannot run schema checks"

    problems = []
    for f in files:
        try:
            cj = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            problems.append(f"{f}: invalid JSON: {e}")
            continue

        # Support files that contain a single card dict or a list of cards
        cards = cj if isinstance(cj, list) else [cj]
        for card in cards:
            for cmd in iter_commands_in_card(card):
                if not isinstance(cmd, dict):
                    continue
                ctype = cmd.get("type")
                if not ctype:
                    problems.append(f"{f}: command missing type: {cmd}")
                    continue
                req = REQUIRED_FIELDS.get(ctype, None)
                if req is None:
                    # Unknown command type — flag as warning but not fail
                    continue
                for key in req:
                    # allow alternate legacy keys
                    aliases = KEY_ALIASES.get(key, [key])
                    found = any((alias in cmd) for alias in aliases)
                    if not found:
                        problems.append(f"{f}: command {ctype} missing required key '{key}' (tried {aliases}): {cmd}")

    assert not problems, "Found saved JSON schema issues:\n" + "\n".join(problems)
