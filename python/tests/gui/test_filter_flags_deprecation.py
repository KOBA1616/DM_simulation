import pytest

from dm_toolkit.gui.editor.models import FilterModel


def test_ingest_legacy_flags_maps_and_serialize_no_flags():
    data = {
        "zones": ["BATTLE_ZONE"],
        "flags": ["tapped", "blocker"],
        "is_evolution": False,
        "min_cost": 2,
    }
    fm = FilterModel.model_validate(data)
    assert fm.is_tapped is True
    assert fm.is_blocker is True
    assert fm.is_evolution is False
    dumped = fm.model_dump()
    # legacy 'flags' must not be emitted after serialization
    assert 'flags' not in dumped
    assert dumped.get('is_tapped') is True
    assert dumped.get('min_cost') == 2


def test_serialize_only_set_fields():
    fm = FilterModel(zones=[], is_tapped=None, is_blocker=True)
    dumped = fm.model_dump()
    # is_tapped unset -> not present
    assert 'is_tapped' not in dumped
    assert dumped['is_blocker'] is True
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
