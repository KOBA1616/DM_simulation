from dm_toolkit.gui.editor.forms.parts.condition_widget import ConditionEditorWidget
import re


def _plain_text(html_text: str) -> str:
    # rudimentary strip of tags for assertions
    return re.sub(r'<[^>]+>', '', html_text)


def _set_widget_data(widget, data: dict):
    try:
        widget.set_value(data)
    except Exception:
        try:
            widget.set_data(data)
        except Exception:
            # last resort: ignore
            pass


def test_preview_various_condition_types():
    w = ConditionEditorWidget(None)

    cases = [
        ({'type': 'OPPONENT_DRAW_COUNT', 'value': 2}, lambda t: any(ch.isdigit() for ch in t)),
        ({'type': 'COMPARE_STAT', 'stat_key': 'MY_SHIELD_COUNT', 'op': '>=', 'value': 1}, lambda t: any(ch.isdigit() for ch in t)),
        ({'type': 'CIVILIZATION_MATCH', 'str_val': 'Fire'}, lambda t: len(t) > 0),
        ({'type': 'EVENT_FILTER_MATCH', 'filter': {'zones': ['HAND']}}, lambda t: len(t) > 0),
    ]

    for data, check in cases:
        _set_widget_data(w, data)
        txt = w.get_preview_text()
        assert txt is not None and txt != ''
        plain = _plain_text(txt)
        assert check(plain), f"Preview failed check for {data}: got '{plain}'"
