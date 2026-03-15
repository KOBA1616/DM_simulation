from dm_toolkit.gui.editor import schema_config
from dm_toolkit import consts


def test_target_scopes_points_to_const_targetscope():
    # schema_config.TARGET_SCOPES should reference TargetScope values from consts
    assert isinstance(schema_config.TARGET_SCOPES, list)
    assert schema_config.TARGET_SCOPES[0] == consts.TargetScope.PLAYER_SELF


def test_duration_options_are_const_duration_types():
    # DURATION_OPTIONS in schema_config should be the canonical DURATION_TYPES from consts
    assert schema_config.DURATION_OPTIONS is consts.DURATION_TYPES or schema_config.DURATION_OPTIONS == consts.DURATION_TYPES
