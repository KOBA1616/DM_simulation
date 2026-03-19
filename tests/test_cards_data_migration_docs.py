# -*- coding: utf-8 -*-
"""Contract test: require a cards data migration procedure document.

This is the RED step: the project should include a migration guide that
describes how to perform data migration for `data/cards.json` including
diff verification steps.
"""
from pathlib import Path


def test_cards_data_migration_doc_exists_and_has_diff_section():
    p = Path("docs/cards_data_migration.md")
    assert p.exists(), "docs/cards_data_migration.md が存在しません。データ移行手順書を追加してください。"
    text = p.read_text(encoding="utf-8")
    assert "差分検証" in text or "diff" in text.lower(), "docs/cards_data_migration.md に差分検証手順が含まれていません。"
