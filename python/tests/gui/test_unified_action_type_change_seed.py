from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class _Item:
    def __init__(self, data):
        self._data = data

    def data(self, role):
        if role == 258:  # Qt.ItemDataRole.UserRole + 2
            return self._data
        return None


class _Widget:
    def __init__(self):
        self.value = None

    def set_value(self, value):
        self.value = value


def test_seed_widgets_from_current_item_keeps_shared_fields() -> None:
    form = UnifiedActionForm(None)
    form.current_item = _Item(
        {
            "type": "QUERY",
            "params": {
                "target_group": "PLAYER_OPPONENT",
                "target_filter": {"zones": ["BATTLE_ZONE"]},
                "str_param": "SELECT_TARGET",
            },
            "input_value_key": "in_key",
            "output_value_key": "out_key",
        }
    )
    form._is_populating = False
    form.widgets_map = {
        "target_group": _Widget(),
        "target_filter": _Widget(),
        "str_param": _Widget(),
        "input_link": _Widget(),
    }

    form._seed_widgets_from_current_item("QUERY")

    assert form.widgets_map["target_group"].value == "PLAYER_OPPONENT"
    assert form.widgets_map["target_filter"].value == {"zones": ["BATTLE_ZONE"]}
    assert form.widgets_map["str_param"].value == "SELECT_TARGET"
    assert form.widgets_map["input_link"].value == {"input_value_key": "in_key", "output_value_key": "out_key"}
