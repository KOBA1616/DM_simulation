"""Tests for EditorConfigLoader single-route config loading behavior."""

import io
import importlib

from dm_toolkit.gui.editor.configs.config_loader import EditorConfigLoader


def test_load_prefers_command_ui_single_route(monkeypatch):
    """RED/GREEN target: command_ui.json is the primary and only route when present."""
    EditorConfigLoader._config_cache = None
    opened_paths = []

    def fake_open(path, mode="r", encoding=None):
        opened_paths.append(path)
        if path == "data/configs/command_ui.json":
            return io.StringIO('{"DRAW": {"visible": []}}')
        raise AssertionError(f"Unexpected fallback path opened: {path}")

    monkeypatch.setattr("builtins.open", fake_open)

    cfg = EditorConfigLoader.load()

    assert "DRAW" in cfg
    assert opened_paths == ["data/configs/command_ui.json"]


def test_load_falls_back_to_legacy_when_command_ui_missing(monkeypatch):
    """When command_ui.json is unavailable, loader should use legacy layout path."""
    EditorConfigLoader._config_cache = None
    opened_paths = []

    def fake_open(path, mode="r", encoding=None):
        opened_paths.append(path)
        if path == "data/configs/command_ui.json":
            raise FileNotFoundError(path)
        if path == "data/editor/editor_layout.json":
            return io.StringIO('{"COMMAND_GROUPS": {"CORE": ["DRAW"]}}')
        raise FileNotFoundError(path)

    monkeypatch.setattr("builtins.open", fake_open)

    cfg = EditorConfigLoader.load()

    assert "COMMAND_GROUPS" in cfg
    assert opened_paths == [
        "data/configs/command_ui.json",
        "data/editor/editor_layout.json",
    ]


def test_command_config_delegates_to_editor_config_loader(monkeypatch):
    """C-2 target: command_config loader should call EditorConfigLoader.load()."""
    expected_cfg = {"DRAW": {"visible": []}}

    def fake_load(cls):
        return expected_cfg

    monkeypatch.setattr(EditorConfigLoader, "load", classmethod(fake_load))

    # Reload to avoid stale module-level state from previous imports.
    import dm_toolkit.gui.editor.forms.command_config as command_config

    command_config = importlib.reload(command_config)
    loaded = command_config.load_command_config()

    assert loaded == expected_cfg
