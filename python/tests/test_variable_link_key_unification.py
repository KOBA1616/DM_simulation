"""Tests for unifying variable link keys to input_value_key/output_value_key."""

from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class StubLinksWidget:
    def __init__(self, payload):
        self._payload = payload

    def get_value(self):
        return self._payload


class StubTypeCombo:
    def currentData(self):
        return "DRAW"


def test_save_normalizes_link_keys_to_standard_names():
    form = UnifiedActionForm(None)
    form.type_combo = StubTypeCombo()

    # Simulate links widget returning legacy keys
    form.widgets_map = {
        'links': StubLinksWidget({'input_var': 'in_foo', 'output_var': 'out_bar'})
    }

    data = {}
    form._save_ui_to_data(data)

    # Expect standardized keys to be present and legacy keys not present
    assert 'input_value_key' in data, f"expected input_value_key, got {data}"
    assert 'output_value_key' in data, f"expected output_value_key, got {data}"
    assert 'input_var' not in data, "legacy key input_var should not be saved"
    assert 'output_var' not in data, "legacy key output_var should not be saved"

    assert data.get('input_value_key') == 'in_foo'
    assert data.get('output_value_key') == 'out_bar'
