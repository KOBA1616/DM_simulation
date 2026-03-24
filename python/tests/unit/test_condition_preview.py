from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget


def test_preview_for_opponent_draw_count():
    w = ConditionEditorWidget()
    data = {"type": "OPPONENT_DRAW_COUNT", "value": 3}
    w.set_data(data)
    preview = w.get_preview_text()
    assert "3枚目" in preview or "3枚目以降" in preview


def test_preview_empty_for_none():
    w = ConditionEditorWidget()
    w.set_data({})
    preview = w.get_preview_text()
    assert preview == "" or "(no preview)" in preview
