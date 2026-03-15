import importlib

from dm_toolkit import consts


def test_schema_config_targets_and_duration_sync():
    schema_cfg = importlib.import_module("dm_toolkit.gui.editor.schema_config")

    # TARGET_SCOPES in schema_config should reference the canonical consts.TARGET_SCOPES
    assert hasattr(schema_cfg, "TARGET_SCOPES")
    assert schema_cfg.TARGET_SCOPES == consts.TARGET_SCOPES

    # DURATION_OPTIONS in schema_config should alias consts.DURATION_TYPES
    assert hasattr(schema_cfg, "DURATION_OPTIONS")
    assert schema_cfg.DURATION_OPTIONS == consts.DURATION_TYPES
# -*- coding: utf-8 -*-
from dm_toolkit.gui.editor import schema_config
from dm_toolkit import consts


def test_target_scopes_match_consts():
    # TARGET_SCOPES should reference TargetScope aliases
    expected = [consts.TargetScope.PLAYER_SELF, consts.TargetScope.PLAYER_OPPONENT, consts.TargetScope.ALL]
    assert schema_config.TARGET_SCOPES == expected


def test_duration_options_match_consts():
    assert schema_config.DURATION_OPTIONS == consts.DURATION_TYPES
