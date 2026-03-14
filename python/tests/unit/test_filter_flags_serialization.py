from dm_toolkit.gui.editor.models import FilterModel


def test_filter_flags_not_serialized_and_mapped():
    data = {
        'zones': ['HAND'],
        'flags': ['tapped', 'blocker']
    }
    f = FilterModel.model_validate(data)
    out = f.model_dump()
    # Legacy 'flags' must not be present in serialized output
    assert 'flags' not in out
    # Mapped explicit fields should be present
    assert out.get('is_tapped') is True
    assert out.get('is_blocker') is True
