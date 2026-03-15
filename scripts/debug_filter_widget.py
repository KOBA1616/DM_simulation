from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.models import FilterSpec
import types

fs = FilterSpec(zones=['HAND','BATTLE_ZONE'], min_cost=2, is_tapped=True, extras={'note':'x'})
w = FilterEditorWidget()
for cb in list(w.zone_checks.values()) + list(w.type_checks.values()):
    if not hasattr(cb, 'setChecked'):
        def _setChecked(self, v):
            setattr(self, '_checked', bool(v))
        def _isChecked(self):
            return getattr(self, '_checked', False)
        cb.setChecked = types.MethodType(_setChecked, cb)
        cb.isChecked = types.MethodType(_isChecked, cb)
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
print('get_data ->', w.get_data())
print('get_filter_spec ->', w.get_filter_spec())
