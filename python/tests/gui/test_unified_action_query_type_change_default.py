from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class _Item:
    def __init__(self, data):
        self._data = data

    def data(self, role):
        if role == 258:
            return self._data
        return None


class _Widget:
    def __init__(self):
        self.value = None

    def set_value(self, value):
        self.value = value


def test_seed_widgets_sets_default_query_mode_when_switching_from_other_type() -> None:
    form = UnifiedActionForm(None)
    form.current_item = _Item(
        {
            "type": "DRAW_CARD",
            "params": {"amount": 2},
        }
    )
    form._is_populating = False
    form.widgets_map = {
        "str_param": _Widget(),
        "amount": _Widget(),
    }

    form._seed_widgets_from_current_item("QUERY")

    assert form.widgets_map["str_param"].value == "CARDS_MATCHING_FILTER"
