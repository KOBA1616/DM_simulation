# -*- coding: utf-8 -*-
import json
from pathlib import Path


def _iter_commands(obj):
    if isinstance(obj, dict):
        if "type" in obj:
            yield obj
        for v in obj.values():
            yield from _iter_commands(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_commands(item)


def test_cards_json_query_commands_have_mode() -> None:
    """再発防止: cards.json の QUERY は mode キー未設定で保存しない。"""
    cards = json.loads(Path("data/cards.json").read_text(encoding="utf-8"))

    missing = []
    for cmd in _iter_commands(cards):
        if str(cmd.get("type", "")).upper() != "QUERY":
            continue
        mode = cmd.get("str_param") or cmd.get("query_mode") or cmd.get("query_string")
        if mode in (None, ""):
            missing.append(cmd.get("uid", "<no-uid>"))

    assert not missing, f"QUERY mode 未設定コマンドがあります: {missing}"
