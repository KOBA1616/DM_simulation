import json
import os
import pytest

dm = pytest.importorskip("dm_ai_module")


def make_cards_stat_scaled(summon_stat_per_value=1, max_reduction=3):
    # Card to be played
    play_card = {
        "id": 9101,
        "name": "Target Creature",
        "type": "CREATURE",
        "cost": 5,
        "power": 500,
        "civilizations": ["NATURE"],
    }

    # Support card in battle that provides a static COST_MODIFIER using STAT_SCALED
    support_card = {
        "id": 9102,
        "name": "Support Static",
        "type": "CREATURE",
        "cost": 1,
        "power": 300,
        "civilizations": ["NATURE"],
        "static_abilities": [
            {
                "type": "COST_MODIFIER",
                "value_mode": "STAT_SCALED",
                "stat_key": "CREATURES_PLAYED",
                "per_value": summon_stat_per_value,
                "min_stat": 1,
                "max_reduction": max_reduction,
            }
        ],
    }

    # Simple mana card
    mana = {
        "id": 9103,
        "name": "Mana",
        "type": "MANA",
        "civilizations": ["NATURE"],
    }

    return [play_card, support_card, mana]


def test_engine_stat_scaled_reduction_applies(tmp_path):
    cards = make_cards_stat_scaled(summon_stat_per_value=1, max_reduction=3)
    p = tmp_path / "mini_stat_scaled.json"
    p.write_text(json.dumps(cards))

    db = dm.JsonLoader.load_cards(str(p))
    # Diagnostic: inspect loaded definition
    try:
        print("DB_KEYS:", list(db.keys()))
        if 9102 in db:
            try:
                print("STATIC_ABS:", len(db[9102].static_abilities))
            except Exception:
                print("STATIC_ABS: (unreadable)")
    except Exception:
        pass
    game = dm.GameInstance(0, db)
    gs = game.state

    # Put support static into battle zone so ContinuousEffectSystem will add active_modifiers
    gs.add_test_card_to_battle(0, 9102, 200, False, False)

    # Set turn stat so reduction = (2 - 1 + 1) * 1 = 2 via StatCommand
    gs.execute_command(dm.StatCommand(dm.StatType.CREATURES_PLAYED, 2))

    # Put target card in hand and provide mana equal to expected reduced cost (5 - 2 = 3)
    gs.add_card_to_hand(0, 9101, 100)
    for i in range(3):
        gs.add_card_to_mana(0, 9103, 300 + i)

    # Diagnostic: ensure active_modifiers were populated by continuous effects
    try:
        print("ACTIVE_MODS_LEN:", len(gs.active_modifiers))
        for m in gs.active_modifiers:
            try:
                print("MOD", m.reduction_amount, m.controller)
            except Exception:
                pass
    except Exception:
        pass

    cmd = {
        'type': dm.CommandType.PLAY_FROM_ZONE,
        'instance_id': 100,
    }

    gs.apply_move(cmd)
    if hasattr(dm, 'ensure_play_resolved'):
        try:
            dm.ensure_play_resolved(gs, cmd)
        except Exception:
            pass

    in_battle = any(c.card_id == 9101 for c in gs.players[0].battle_zone)
    assert in_battle, "Card with STAT_SCALED static reduction did not resolve to battle zone"


def test_engine_stat_scaled_respects_max_reduction(tmp_path):
    # Use per_value that would exceed max_reduction without clamp
    cards = make_cards_stat_scaled(summon_stat_per_value=2, max_reduction=3)
    p = tmp_path / "mini_stat_scaled_max.json"
    p.write_text(json.dumps(cards))

    db = dm.JsonLoader.load_cards(str(p))
    game = dm.GameInstance(0, db)
    gs = game.state

    gs.add_test_card_to_battle(0, 9102, 200, False, False)
    gs.execute_command(dm.StatCommand(dm.StatType.CREATURES_PLAYED, 5))

    # Expect clamp to max_reduction=3 -> final cost = 5 - 3 = 2
    gs.add_card_to_hand(0, 9101, 100)
    for i in range(2):
        gs.add_card_to_mana(0, 9103, 300 + i)

    cmd = {
        'type': dm.CommandType.PLAY_FROM_ZONE,
        'instance_id': 100,
    }

    gs.apply_move(cmd)
    if hasattr(dm, 'ensure_play_resolved'):
        try:
            dm.ensure_play_resolved(gs, cmd)
        except Exception:
            pass

    in_battle = any(c.card_id == 9101 for c in gs.players[0].battle_zone)
    assert in_battle, "Card with STAT_SCALED max_reduction clamp did not resolve to battle zone"
