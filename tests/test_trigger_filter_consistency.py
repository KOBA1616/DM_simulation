"""Consistency checks for Trigger Scope/Filter in form data."""
from dm_toolkit.gui.editor.consistency import validate_trigger_scope_filter


def test_owner_duplication_warning():
    effect = {
        "trigger": "ON_PLAY",
        "trigger_scope": "OPPONENT",
        "trigger_filter": {"owner": "OPPONENT", "types": ["CREATURE"], "zones": ["BATTLE_ZONE"]},
    }
    warns = validate_trigger_scope_filter(effect)
    assert any("重複" in w for w in warns)


def test_exact_cost_conflict():
    effect = {
        "trigger": "ON_CAST_SPELL",
        "trigger_scope": "SELF",
        "trigger_filter": {"types": ["SPELL"], "exact_cost": 3, "min_cost": 1},
    }
    warns = validate_trigger_scope_filter(effect)
    assert any("競合" in w for w in warns)


def test_zone_type_mismatch_shield():
    effect = {
        "trigger": "ON_SHIELD_ADD",
        "trigger_scope": "SELF",
        "trigger_filter": {"zones": ["SHIELD_ZONE"], "types": ["CREATURE"]},
    }
    warns = validate_trigger_scope_filter(effect)
    assert any("不整合" in w for w in warns)
