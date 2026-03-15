from dm_toolkit.gui.editor.forms.parts.filter_widget import FilterEditorWidget
from dm_toolkit.gui.editor.models import FilterSpec, dict_to_filterspec, filterspec_to_dict


def test_filter_widget_roundtrip():
    # Construct a representative FilterSpec
    fs = FilterSpec(
        zones=["DECK", "HAND"],
        civilizations=["FIRE"],
        races=["ドラゴン"],
        min_cost=1,
        max_cost=5,
        min_power=1000,
        max_power=5000,
        owner="SELF",
        is_tapped=True,
        is_blocker=False,
        extras={"custom_flag": 42}
    )

    w = FilterEditorWidget()
    # The test environment provides lightweight Qt stubs where EnhancedCheckBox
    # may not implement `setChecked`. Monkeypatch instances to accept calls.
    # Patch any checkbox-like stub that provides `isChecked` but lacks `setChecked`.
    for name in dir(w):
        try:
            obj = getattr(w, name)
        except Exception:
            continue
        if hasattr(obj, 'isChecked') and not hasattr(obj, 'setChecked'):
            # store checked state on the instance and provide accessors
            setattr(obj, '_checked', False)
            def _set_checked(v, _obj=obj):
                setattr(_obj, '_checked', bool(v))
            def _is_checked(_obj=obj):
                return bool(getattr(_obj, '_checked', False))
            setattr(obj, 'setChecked', _set_checked)
            setattr(obj, 'isChecked', _is_checked)
    # Also patch any checkboxes stored inside dicts on the widget (e.g., zone_checks)
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

    # Populate widget from FilterSpec
    w.set_filter_spec(fs)

    # Read back as FilterSpec and assert key fields preserved
    out_fs = w.get_filter_spec()
    assert isinstance(out_fs, FilterSpec)
    assert set(out_fs.zones) == set(fs.zones)
    assert out_fs.min_cost == fs.min_cost
    assert out_fs.max_power == fs.max_power
    assert out_fs.is_tapped == fs.is_tapped
    # Round-trip through dict conversion
    d = filterspec_to_dict(out_fs)
    fs2 = dict_to_filterspec(d)
    assert fs2.min_cost == fs.min_cost
    # Note: arbitrary extras may be dropped by the UI widget round-trip; ensure no crash
    assert isinstance(fs2.extras, dict)
