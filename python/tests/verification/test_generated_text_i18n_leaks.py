# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from pathlib import Path

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


_ENUM_TOKEN_RE = re.compile(r"\b[A-Z0-9]+(?:_[A-Z0-9]+)+\b")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_cards(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    if isinstance(data, dict) and "cards" in data and isinstance(data["cards"], list):
        return [c for c in data["cards"] if isinstance(c, dict)]
    raise ValueError(f"Unsupported card JSON shape: {path}")


def _find_enum_tokens(text: str) -> list[str]:
    # We intentionally focus on underscore-style enum tokens to avoid false positives
    # from English card names/races.
    return sorted(set(_ENUM_TOKEN_RE.findall(text)))


def test_generated_text_does_not_leak_enum_tokens_in_test_cards() -> None:
    root = _repo_root()
    cards_path = root / "data" / "test_cards.json"
    cards = _load_cards(cards_path)

    offenders: list[str] = []

    for card in cards:
        cid = card.get("id")
        name = card.get("name")
        text = CardTextGenerator.generate_text(card)

        leaked = _find_enum_tokens(text)
        if leaked:
            offenders.append(f"id={cid} name={name} leaked={leaked}")

    if offenders:
        raise AssertionError(
            "Generated card text leaked enum-like tokens (underscore identifiers). "
            "Add translations or improve formatting:\n" + "\n".join(f"- {x}" for x in offenders)
        )
