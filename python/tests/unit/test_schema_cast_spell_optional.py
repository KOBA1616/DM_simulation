# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.schema_config import register_all_schemas
from dm_toolkit.gui.editor.schema_def import SCHEMA_REGISTRY, get_schema


def test_cast_spell_schema_includes_optional_field():
    SCHEMA_REGISTRY.clear()
    register_all_schemas()

    schema = get_schema("CAST_SPELL")
    assert schema is not None

    keys = [field.key for field in schema.fields]
    assert "optional" in keys
