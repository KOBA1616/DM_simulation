# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor import schema_config, schema_def
from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader


def _reload_schemas() -> None:
    schema_def.SCHEMA_REGISTRY.clear()
    schema_def.COMMAND_REGISTRY.clear()
    EditorConfigLoader._config_cache = None
    schema_config.register_all_schemas()


def test_command_registry_has_required_spec_fields():
    _reload_schemas()
    snap = schema_def.get_command_registry_snapshot()
    assert snap, 'COMMAND_REGISTRY should not be empty after schema registration'

    required = {'type', 'group', 'fields', 'validator', 'text_hint'}
    for cmd, entry in snap.items():
        assert required.issubset(entry.keys()), f'missing keys for {cmd}: {required - set(entry.keys())}'
        assert entry['type'] == cmd
        assert isinstance(entry['group'], str)
        assert isinstance(entry['fields'], list)
        assert isinstance(entry['validator'], str)
        assert isinstance(entry['text_hint'], str)


def test_command_registry_command_set_matches_expected_baseline():
    _reload_schemas()
    # CI diff detection: keep schema-registry and command_ui command sets aligned.
    # Known deltas are intentionally tracked here to detect unexpected drift.
    registered = set(schema_def.get_registered_command_types())
    configured = set(EditorConfigLoader.get_command_ui_config().keys())

    only_in_registry = registered - configured
    only_in_config = configured - registered

    # Current intentional deltas (2026-03-15):
    # - LOCK_SPELL / SELECT_TARGET are schema-registered but not in command_ui config map.
    # - MEASURE_COUNT is present in command_ui config but not schema-registered.
    assert only_in_registry == {'LOCK_SPELL', 'SELECT_TARGET'}
    assert only_in_config == {'MEASURE_COUNT'}


def test_command_registry_and_schema_registry_are_synchronized():
    _reload_schemas()
    schema_types = sorted(schema_def.SCHEMA_REGISTRY.keys())
    registry_types = schema_def.get_registered_command_types()
    assert registry_types == schema_types
