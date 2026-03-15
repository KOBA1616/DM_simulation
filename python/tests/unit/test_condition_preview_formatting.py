from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget


def test_condition_preview_html_and_truncation():
    w = ConditionEditorWidget(None)

    # Simple known type should render with bolded type label
    w.set_data({'type': 'OPPONENT_DRAW_COUNT', 'value': 3})
    txt = w.get_preview_text()
    assert txt is not None
    assert '<b>' in txt and '</b>' in txt

    # Long custom string should be truncated to a reasonable length
    long_val = 'X' * 400
    w.set_data({'type': 'CUSTOM', 'str_val': long_val})
    txt2 = w.get_preview_text()
    assert txt2 is not None
    # Expect visible truncation (ellipsis) and limited length
    assert '…' in txt2 or len(txt2) <= 160
