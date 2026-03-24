from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dm_ai_module as dm


def _entries(card_map: Any) -> list[object]:
    # 再発防止: ネイティブ/フォールバックで返却コンテナ型が異なるため、
    # ここは Any で受けてランタイム分岐し、型チェッカ誤検知を防ぐ。
    values_fn = getattr(card_map, "values", None)
    if callable(values_fn):
        return list(values_fn())

    get_all_cards_fn = getattr(card_map, "get_all_cards", None)
    if callable(get_all_cards_fn):
        try:
            all_cards = get_all_cards_fn()
            all_values_fn = getattr(all_cards, "values", None)
            if callable(all_values_fn):
                return list(all_values_fn())
        except Exception:
            return []

    try:
        return list(card_map)
    except Exception:
        return []


def _write_cards(tmp_path: Path, cards: list[dict]) -> Path:
    path = tmp_path / "cards.json"
    path.write_text(json.dumps(cards), encoding="utf-8")
    return path


def test_legacy_actions_without_schema_version_are_accepted(tmp_path: Path) -> None:
    cards = [
        {
            "id": 9101,
            "name": "legacy-ok",
            "civilizations": [],
            "type": 0,
            "cost": 1,
            "power": 0,
            "races": [],
            "effects": [
                {
                    "trigger": 0,
                    "condition": None,
                    "actions": [
                        {"type": 0, "scope": "SINGLE", "filter": "", "value1": 1, "optional": False}
                    ],
                }
            ],
        }
    ]
    path = _write_cards(tmp_path, cards)

    card_map = dm.JsonLoader.load_cards(str(path))
    entries = _entries(card_map)

    assert len(entries) == 1


def test_schema_v2_card_without_actions_is_accepted(tmp_path: Path) -> None:
    cards = [
        {
            "schema_version": 2,
            "id": 9102,
            "name": "v2-ok",
            "civilizations": [],
            "type": 0,
            "cost": 1,
            "power": 0,
            "races": [],
            "effects": [
                {
                    "trigger": 0,
                    "condition": None,
                    "commands": [],
                }
            ],
        }
    ]
    path = _write_cards(tmp_path, cards)

    card_map = dm.JsonLoader.load_cards(str(path))
    entries = _entries(card_map)

    assert len(entries) == 1


def test_schema_v2_card_with_legacy_actions_is_rejected(tmp_path: Path) -> None:
    cards = [
        {
            "schema_version": 2,
            "id": 9103,
            "name": "v2-reject-legacy-actions",
            "civilizations": [],
            "type": 0,
            "cost": 1,
            "power": 0,
            "races": [],
            "effects": [
                {
                    "trigger": 0,
                    "condition": None,
                    "actions": [
                        {"type": 0, "scope": "SINGLE", "filter": "", "value1": 1, "optional": False}
                    ],
                }
            ],
        }
    ]
    path = _write_cards(tmp_path, cards)

    card_map = dm.JsonLoader.load_cards(str(path))
    entries = _entries(card_map)

    assert entries == []
