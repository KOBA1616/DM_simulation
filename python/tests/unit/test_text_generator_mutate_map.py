from dm_toolkit.gui.editor.formatters.special_effect_formatters import MutateFormatter
from dm_toolkit.gui.editor.formatters.context import TextGenerationContext


def test_mutate_tap():
    action = {
        "mutation_kind": "TAP"
    }
    ctx = TextGenerationContext(card_data={})
    res = MutateFormatter._mutate_tap("対象", 1, "体", "", "", False)
    assert "タップする" in res


def test_mutate_add_keyword():
    action = {
        "mutation_kind": "ADD_KEYWORD",
        "str_val": "CANNOT_ATTACK"
    }
    ctx = TextGenerationContext(card_data={})
    res = MutateFormatter._mutate_add_keyword("対象", 1, "体", "", "CANNOT_ATTACK", False)
    # Ensure output is generated (non-empty) — exact phrasing may vary by keyword
    assert res and len(res.strip()) > 0
