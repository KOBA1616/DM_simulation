from pathlib import Path


def test_cards_data_migration_doc_exists_and_has_header():
    p = Path("docs/cards_data_migration.md")
    assert p.exists(), "Expected docs/cards_data_migration.md to exist"
    text = p.read_text(encoding="utf-8")
    assert ("カードデータ移行" in text) or ("差分検証" in text), "Document should contain migration/diff verification header"
