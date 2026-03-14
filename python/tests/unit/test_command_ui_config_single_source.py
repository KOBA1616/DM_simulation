import json
from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader
from dm_toolkit.gui.editor.forms.command_config import COMMAND_UI_CONFIG


def test_command_ui_config_matches_json_source():
    # Load JSON file directly
    with open('data/configs/command_ui.json', 'r', encoding='utf-8') as f:
        js = json.load(f)
    # EditorConfigLoader should return a mapping of commands
    loader_cfg = EditorConfigLoader.get_command_ui_config()
    # COMMAND_UI_CONFIG built from loader should have same keys as JSON-derived mapping
    json_keys = {k for k, v in js.items() if k.isupper() and isinstance(v, dict)}
    loader_keys = set(loader_cfg.keys())
    command_keys = set(COMMAND_UI_CONFIG.keys())
    assert json_keys == loader_keys
    assert loader_keys == command_keys
