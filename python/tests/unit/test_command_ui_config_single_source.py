import json
from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG


def test_command_ui_config_matches_json_source():
    # Use EditorConfigLoader as single source; COMMAND_UI_CONFIG is built from it
    loader_cfg = EditorConfigLoader.get_command_ui_config()
    loader_keys = set(loader_cfg.keys())
    command_keys = set(COMMAND_UI_CONFIG.keys())
    assert loader_keys == command_keys
