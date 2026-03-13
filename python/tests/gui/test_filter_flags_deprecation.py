# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor.models import FilterModel


def test_filter_flags_are_mapped_and_not_serialized():
    legacy = {
        'zones': ['HAND'],
        'flags': ['is_tapped', 'is_blocker']
    }

    f = FilterModel.model_validate(legacy)
    assert f.is_tapped is True
    assert f.is_blocker is True

    dumped = f.model_dump()
    # legacy 'flags' should not be present in serialized output
    assert 'flags' not in dumped
    # explicit fields should be present
    assert dumped.get('is_tapped') is True
    assert dumped.get('is_blocker') is True
