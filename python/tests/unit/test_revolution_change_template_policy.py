# -*- coding: utf-8 -*-

import json
import pathlib


def test_revolution_change_template_uses_effect_trigger_plus_command_combo():
    root = pathlib.Path(__file__).resolve().parents[3]
    templates = json.loads((root / "data" / "editor_templates.json").read_text(encoding="utf-8"))

    tpl = templates["REVOLUTION_CHANGE"]
    data = tpl["data"]
    commands = data.get("commands", [])

    assert data.get("trigger") == "ON_ATTACK"
    assert data.get("trigger_scope") == "PLAYER_SELF"
    assert len(commands) == 1
    assert commands[0].get("type") == "REVOLUTION_CHANGE"
    assert "mutation_kind" not in commands[0]
    # Regression guard: RC template must expose editable civ/race/cost condition slots.
    target_filter = commands[0].get("target_filter", {})
    assert target_filter.get("civilizations") == []
    assert target_filter.get("races") == []
    assert target_filter.get("min_cost") == -1
    assert target_filter.get("max_cost") == -1
