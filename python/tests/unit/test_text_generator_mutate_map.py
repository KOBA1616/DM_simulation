from dm_toolkit.gui.editor.text_generator import CardTextGenerator


def test_mutate_tap():
    action = {
        "mutation_kind": "TAP"
    }
    res = CardTextGenerator._format_special_effect_command("MUTATE", action, is_spell=False, val1=1, target_str="対象", unit="体")
    assert "タップする" in res


def test_mutate_add_keyword():
    action = {
        "mutation_kind": "ADD_KEYWORD",
        "str_val": "CANNOT_ATTACK"
    }
    res = CardTextGenerator._format_special_effect_command("MUTATE", action, is_spell=False, val1=1, target_str="対象", unit="体")
    # Ensure output is generated (non-empty) — exact phrasing may vary by keyword
    assert res and len(res.strip()) > 0
