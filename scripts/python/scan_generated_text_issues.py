# -*- coding: utf-8 -*-
"""Scan card text generation output for likely unsupported/unlocalized artifacts.

This is a developer tool for the Card Editor / text generator.

Usage (PowerShell):
    & ./.venv/Scripts/python.exe ./scripts/python/scan_generated_text_issues.py --cards data/test_cards.json
    & ./.venv/Scripts/python.exe ./scripts/python/scan_generated_text_issues.py --cards data/cards.json --max 200

Exit code:
  0 if no issues detected
  1 if issues detected
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dm_toolkit.gui.editor.text_generator import CardTextGenerator


_ENUM_TOKEN_RE = re.compile(r"\b[A-Z0-9]+(?:_[A-Z0-9]+)+\b")


def _load_cards(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [c for c in data if isinstance(c, dict)]
    if isinstance(data, dict) and "cards" in data and isinstance(data["cards"], list):
        return [c for c in data["cards"] if isinstance(c, dict)]
    raise ValueError(f"Unsupported card JSON shape: {path}")


def _find_issues(text: str) -> list[str]:
    issues: list[str] = []

    # Common fallback signature: "(SOME_ENUM)" produced by CardTextGenerator when template missing.
    if re.search(r"\([A-Z0-9_]{3,}\)", text):
        issues.append("fallback_paren_enum")

    # Enum-like tokens with underscores leaked into the final text.
    leaked = sorted(set(_ENUM_TOKEN_RE.findall(text)))
    if leaked:
        issues.append("enum_token_leak:" + ",".join(leaked[:10]) + ("..." if len(leaked) > 10 else ""))

    # MUTATE fallback may include raw mutation kind.
    if re.search(r"状態変更\([^\)]*[A-Z0-9_]{3,}[^\)]*\)", text):
        issues.append("mutate_kind_leak")

    # Trigger fallback may include raw trigger tokens.
    if re.search(r"\bON_[A-Z0-9_]+\b|\bAT_[A-Z0-9_]+\b", text):
        issues.append("trigger_token_leak")

    # English UI-ish words in what should be rule text (heuristic).
    if re.search(r"\b(Unknown|Unsupported|invalid|error)\b", text):
        issues.append("english_word_leak")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", default="data/test_cards.json")
    ap.add_argument("--max", type=int, default=0, help="Max cards to scan (0 = all)")
    args = ap.parse_args()

    path = Path(args.cards)
    cards = _load_cards(path)
    if args.max and args.max > 0:
        cards = cards[: args.max]

    findings: list[tuple[int | None, str, list[str]]] = []

    for card in cards:
        cid = card.get("id")
        name = str(card.get("name") or "")
        try:
            text = CardTextGenerator.generate_text(card)
        except Exception as e:
            findings.append((cid, name, [f"generator_exception:{type(e).__name__}:{e}"]))
            continue

        issues = _find_issues(text)
        if issues:
            findings.append((cid, name, issues))

    if not findings:
        print("OK: no suspicious artifacts detected")
        return 0

    print(f"Found {len(findings)} cards with suspicious generated-text artifacts")
    for cid, name, issues in findings[:200]:
        label = f"id={cid}" if cid is not None else "id=?"
        if name:
            label += f" name={name}"
        print(f"- {label}\n  - " + "\n  - ".join(issues))

    if len(findings) > 200:
        print(f"... and {len(findings) - 200} more")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
