# -*- coding: utf-8 -*-

from dm_toolkit.gui.editor.consistency import validate_command_list


def test_transition_source_zone_filter_zone_conflict_warns():
    commands = [
        {
            "type": "TRANSITION",
            "from_zone": "HAND",
            "to_zone": "GRAVEYARD",
            "target_filter": {"zones": ["BATTLE_ZONE"], "types": ["CREATURE"]},
            "amount": 1,
        }
    ]

    warns = validate_command_list(commands, _path="TRANSITION")

    assert any("競合" in w and "Source Zone=HAND" in w for w in warns)


def test_transition_source_zone_filter_zone_match_no_warning():
    commands = [
        {
            "type": "TRANSITION",
            "from_zone": "HAND",
            "to_zone": "GRAVEYARD",
            "target_filter": {"zones": ["HAND"], "types": ["CREATURE"]},
            "amount": 1,
        }
    ]

    warns = validate_command_list(commands, _path="TRANSITION")

    assert not any("Source Zone=" in w for w in warns)


def test_transition_without_filter_zones_does_not_warn():
    commands = [
        {
            "type": "TRANSITION",
            "from_zone": "HAND",
            "to_zone": "GRAVEYARD",
            "target_filter": {"types": ["CREATURE"]},
            "amount": 1,
        }
    ]

    warns = validate_command_list(commands, _path="TRANSITION")

    assert not any("Source Zone=" in w for w in warns)
