import pytest

from dm_toolkit.gui.editor.forms.unified_action_form import UnifiedActionForm


class StubCombo:
    def currentData(self):
        return "QUERY"


def test_save_blocks_invalid_query_select_option():
    """RED: Saving a QUERY without required params should be blocked by validation."""
    form = UnifiedActionForm(None)

    # Replace the combo with a stub that reports QUERY
    form.type_combo = StubCombo()

    data = {}

    # Ensure widgets_map is empty to simulate missing str_param / target_filter
    form.widgets_map = {}

    form._save_ui_to_data(data)

    # Expect: save aborted and data remains empty
    assert data == {}, "Invalid QUERY should not be saved (data must remain empty)"
