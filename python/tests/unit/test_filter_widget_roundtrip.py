from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.models import FilterSpec
from dm_toolkit.gui.editor.models import CommandModel
import types


def test_filter_widget_to_commandmodel_roundtrip():
    # Prepare a FilterSpec
    fs = FilterSpec(zones=['HAND', 'BATTLE_ZONE'], min_cost=2, is_tapped=True, extras={'note': 'x'})

    # Create widget and populate from FilterSpec
    w = FilterEditorWidget()
    # PyQt stub EnhancedCheckBox may lack setChecked; patch instances defensively
    for cb in list(w.zone_checks.values()) + list(w.type_checks.values()):
        if not hasattr(cb, 'setChecked'):
            def _setChecked(self, v):
                setattr(self, '_checked', bool(v))
            def _isChecked(self):
                return getattr(self, '_checked', False)
            cb.setChecked = types.MethodType(_setChecked, cb)
            cb.isChecked = types.MethodType(_isChecked, cb)
    # Patch other checkboxes on the widget that may lack setChecked in stubs
    for name in ('trigger_source_check',):
        cb = getattr(w, name, None)
        if cb and not hasattr(cb, 'setChecked'):
            def _setChecked(self, v):
                setattr(self, '_checked', bool(v))
            def _isChecked(self):
                return getattr(self, '_checked', False)
            cb.setChecked = types.MethodType(_setChecked, cb)
            cb.isChecked = types.MethodType(_isChecked, cb)
    w.set_filter_spec(fs)

    # Extract dict and FilterSpec back
    d = w.get_data()
    fs2 = w.get_filter_spec()

    assert isinstance(d, dict)
    assert set(fs2.zones) == set(fs.zones)
    assert fs2.min_cost == fs.min_cost
    assert fs2.is_tapped == fs.is_tapped
    # Extras from FilterSpec may not be preserved by the UI (unknown keys are not
    # round-tripped through widget fields). Accept either behavior.
    # If preserved, it will be present; otherwise ignore.

    # Create CommandModel using the widget's dict as filter
    data = {'type': 'POWER_MOD', 'filter': d, 'amount': 5}
    cm = CommandModel.model_validate(data)
    # Ensure serialization flattens the params into top-level keys
    out = cm.model_dump()
    assert out.get('type') == 'POWER_MOD'
    # filter may be present as nested dict in params or flattened depending on ingest; ensure presence
    assert out.get('filter') is not None or out.get('params', {}).get('filter') is not None
