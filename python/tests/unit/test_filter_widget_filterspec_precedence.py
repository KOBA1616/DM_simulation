import warnings
from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.models import FilterSpec


def _patch_widget_checkboxes(w):
    # Patch any checkbox-like stub that provides `isChecked` but lacks `setChecked`.
    for name in dir(w):
        try:
            obj = getattr(w, name)
        except Exception:
            continue
        if hasattr(obj, 'isChecked') and not hasattr(obj, 'setChecked'):
            setattr(obj, '_checked', False)
            def _set_checked(v, _obj=obj):
                setattr(_obj, '_checked', bool(v))
            def _is_checked(_obj=obj):
                return bool(getattr(_obj, '_checked', False))
            setattr(obj, 'setChecked', _set_checked)
            setattr(obj, 'isChecked', _is_checked)
    try:
        for cb in list(getattr(w, 'zone_checks', {}).values()) + list(getattr(w, 'type_checks', {}).values()):
            if hasattr(cb, 'isChecked') and not hasattr(cb, 'setChecked'):
                setattr(cb, '_checked', False)
                def _set_checked_cb(v, _cb=cb):
                    setattr(_cb, '_checked', bool(v))
                def _is_checked_cb(_cb=cb):
                    return bool(getattr(_cb, '_checked', False))
                setattr(cb, 'setChecked', _set_checked_cb)
                setattr(cb, 'isChecked', _is_checked_cb)
    except Exception:
        pass


def test_set_data_with_filterspec_no_warning():
    w = FilterEditorWidget()
    _patch_widget_checkboxes(w)
    fs = FilterSpec(zones=["HAND"], civilizations=["FIRE"]) 
    # should not warn
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w.set_data(fs)
        assert not any(issubclass(r.category, DeprecationWarning) for r in rec)


def test_set_data_with_dict_emits_deprecation():
    w = FilterEditorWidget()
    _patch_widget_checkboxes(w)
    d = {"zones": ["HAND"]}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        w.set_data(d)
        assert any(issubclass(r.category, DeprecationWarning) for r in rec)
