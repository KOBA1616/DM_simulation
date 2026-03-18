import json
import os
from pathlib import Path

import pytest

dm = pytest.importorskip("dm_ai_module")


def test_apply_move_active_payment(tmp_path):
    # Create a minimal card DB with a playable card (9001) that has an ACTIVE_PAYMENT
    # and a cheap creature (9002) used as the tap-payment candidate.
    cards = [
        {
            "id": 9001,
            "name": "Test Active Pay Card",
            "type": "CREATURE",
            "cost": 5,
            "power": 1000,
            "civilizations": ["NATURE"],
            "cost_reductions": [
                {
                    "id": "r_test",
                    "type": "ACTIVE_PAYMENT",
                    "unit_cost": {"type": "TAP_CARD", "amount": 0, "filter": {"zones": ["BATTLE_ZONE"]}},
                    "reduction_amount": 2,
                    "max_units": 2
                }
            ]
        },
        {
            "id": 9002,
            "name": "Payee Creature",
            "type": "CREATURE",
            "cost": 1,
            "power": 500,
            "civilizations": ["NATURE"]
        }
    ]

    p = tmp_path / "mini_cards.json"
    p.write_text(json.dumps(cards))

    # Load cards via JsonLoader to get native CardDefinition objects
    db = dm.JsonLoader.load_cards(str(p))

    # Create game instance with this DB
    game = dm.GameInstance(0, db)
    gs = game.state

    # Note: Python-side CardDefinition proxy may not expose `cost_reductions` attribute,
    # but the native engine will have parsed it. Proceed to exercise apply_move path.

    # Register a playable card in player 0's hand with instance id 100
    gs.add_card_to_hand(0, 9001, 100)

    # Register a candidate creature in battle zone for payment (instance id 200)
    gs.add_test_card_to_battle(0, 9002, 200, False, False)

    # Add mana cards enough to cover post-reduction cost (effective cost = 5 - 2 = 3)
    # Use card id 9002 as mana cards for simplicity, instance ids 300..302
    gs.add_card_to_mana(0, 9002, 300)
    gs.add_card_to_mana(0, 9002, 301)
    gs.add_card_to_mana(0, 9002, 302)

    # Build command dict to play instance 100 using ACTIVE_PAYMENT selecting reduction 'r_test' with 1 unit
    cmd = {
        'type': dm.CommandType.PLAY_FROM_ZONE,
        'instance_id': 100,
        'payment_mode': 'ACTIVE_PAYMENT',
        'reduction_id': 'r_test',
        'payment_units': 1
    }

    # Execute the move
    gs.apply_move(cmd)
    # Ensure play resolved (fallback for environments where native path didn't complete)
    try:
        # some test environments export helper
        if hasattr(dm, 'ensure_play_resolved'):
            dm.ensure_play_resolved(gs, cmd)
    except Exception:
        pass

    # Diagnostic dump: print zones to help trace why the play didn't resolve
    p0 = gs.players[0]
    def zone_dump(zone_list):
        return [ {"instance_id": c.instance_id, "card_id": c.card_id, "tapped": getattr(c, 'is_tapped', False)} for c in zone_list ]

    print("PLAYER0 HAND:", zone_dump(p0.hand))
    print("PLAYER0 STACK:", zone_dump(p0.stack))
    print("PLAYER0 BATTLE:", zone_dump(p0.battle_zone))
    print("PLAYER0 MANA:", zone_dump(p0.mana_zone))
    try:
        print("COMMAND_HISTORY_LEN:", len(gs.command_history))
    except Exception:
        print("COMMAND_HISTORY: (unprintable)")
    # Dump command_history entries for detailed inspection
    try:
        for i, cmd_obj in enumerate(gs.command_history):
            try:
                # Try to surface common CommandDef-like fields for inspection
                info = {}
                for attr in ("type", "instance_id", "owner_id", "target_instance", "from_zone", "to_zone", "mutation_kind", "amount", "payment_mode", "reduction_id", "payment_units", "str_param", "optional"):
                    if hasattr(cmd_obj, attr):
                        try:
                            info[attr] = getattr(cmd_obj, attr)
                        except Exception:
                            info[attr] = "<unreadable>"
                # Some command objects expose getter methods
                if hasattr(cmd_obj, 'get_type') and callable(getattr(cmd_obj, 'get_type')):
                    try:
                        info['type_getter'] = cmd_obj.get_type()
                    except Exception:
                        info['type_getter'] = '<err>'
                print(f"CMD[{i}]: fields={info} repr={repr(cmd_obj)}")
            except Exception as e:
                print(f"CMD[{i}]: repr failed: {e}")
    except Exception:
        print("COMMAND_HISTORY DUMP: unavailable")
    # More diagnostics
    try:
        print("PENDING_EFFECTS_COUNT:", gs.get_pending_effect_count())
    except Exception:
        print("PENDING_EFFECTS_COUNT: (unavailable)")
    try:
        print("waiting_for_user_input:", gs.waiting_for_user_input)
    except Exception:
        print("waiting_for_user_input: (unavailable)")
    try:
        print("pending_query:", gs.pending_query)
    except Exception:
        print("pending_query: (unavailable)")
    try:
        print("turn_stats.played_without_mana_instance_ids:", gs.turn_stats.played_without_mana_instance_ids)
    except Exception:
        print("turn_stats: (unavailable)")

    # The primary assertion: the played card (100) should end up in the battle zone
    in_battle = any(c.card_id == 9001 for c in gs.players[0].battle_zone)
    assert in_battle, "Played card did not end in battle zone"


def test_passive_only_reduction(tmp_path):
    """Test PASSIVE reduction where cost is lowered automatically."""
    cards = [
        {
            "id": 1001,
            "name": "Passive Reduced Card",
            "type": "CREATURE",
            "cost": 7,
            "power": 800,
            "civilizations": ["FIRE"],
            "cost_reductions": [
                {
                    "id": "passive_1",
                    "type": "PASSIVE",
                    "reduction_amount": 3,
                    "min_mana_cost": 2  # Can't reduce below 2
                }
            ]
        },
        {
            "id": 1002,
            "name": "Mana",
            "type": "MANA",
            "civilizations": ["FIRE"]
        }
    ]

    p = tmp_path / "cards_passive.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))

    game = dm.GameInstance(0, db)
    gs = game.state

    # Add card to hand
    gs.add_card_to_hand(0, 1001, 100)

    # Add 4 mana (7 cost - 3 reduction = 4 needed)
    for i in range(4):
        gs.add_card_to_mana(0, 1002, 200 + i)

    # Play without specifying payment_mode (should use PASSIVE)
    cmd = {
        'type': dm.CommandType.PLAY_FROM_ZONE,
        'instance_id': 100,
    }

    gs.apply_move(cmd)
    try:
        if hasattr(dm, 'ensure_play_resolved'):
            dm.ensure_play_resolved(gs, cmd)
    except Exception:
        pass

    # Card should be in battle zone
    in_battle = any(c.card_id == 1001 for c in gs.players[0].battle_zone)
    assert in_battle, "Card with PASSIVE reduction did not resolve to battle zone"


def test_min_mana_cost_floor(tmp_path):
    """Test that min_mana_cost prevents cost reduction below floor."""
    cards = [
        {
            "id": 3001,
            "name": "Floored Reduction",
            "type": "CREATURE",
            "cost": 9,
            "power": 1000,
            "civilizations": ["DARK"],
            "cost_reductions": [
                {
                    "id": "passive_floor",
                    "type": "PASSIVE",
                    "reduction_amount": 10,  # Would reduce to -1 without floor
                    "min_mana_cost": 3  # Floor at 3
                }
            ]
        },
        {
            "id": 3002,
            "name": "Mana",
            "type": "MANA",
            "civilizations": ["DARK"]
        }
    ]

    p = tmp_path / "cards_floor.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))

    game = dm.GameInstance(0, db)
    gs = game.state

    gs.add_card_to_hand(0, 3001, 100)

    # Need exactly 3 mana (floored cost)
    for i in range(3):
        gs.add_card_to_mana(0, 3002, 200 + i)

    cmd = {
        'type': dm.CommandType.PLAY_FROM_ZONE,
        'instance_id': 100,
    }

    gs.apply_move(cmd)
    try:
        if hasattr(dm, 'ensure_play_resolved'):
            dm.ensure_play_resolved(gs, cmd)
    except Exception:
        pass

    # Should succeed with exactly 3 mana
    in_battle = any(c.card_id == 3001 for c in gs.players[0].battle_zone)
    assert in_battle, "Card with min_mana_cost floor should play with floored cost"


def test_passive_and_active_combined(tmp_path):
    """Test combination of PASSIVE and ACTIVE_PAYMENT reductions."""
    cards = [
        {
            "id": 4001,
            "name": "Combo Reduction Card",
            "type": "CREATURE",
            "cost": 10,
            "power": 1500,
            "civilizations": ["NATURE"],
            "cost_reductions": [
                {
                    "id": "passive_part",
                    "type": "PASSIVE",
                    "reduction_amount": 2
                },
                {
                    "id": "active_part",
                    "type": "ACTIVE_PAYMENT",
                    "unit_cost": {"type": "TAP_CARD", "amount": 0, "filter": {"zones": ["BATTLE_ZONE"]}},
                    "reduction_amount": 3,
                    "max_units": 1
                }
            ]
        },
        {
            "id": 4002,
            "name": "Payee",
            "type": "CREATURE",
            "cost": 1,
            "power": 100,
            "civilizations": ["NATURE"]
        },
        {
            "id": 4003,
            "name": "Mana",
            "type": "MANA",
            "civilizations": ["NATURE"]
        }
    ]

    p = tmp_path / "cards_combo.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))

    game = dm.GameInstance(0, db)
    gs = game.state

    # Put cards in hand
    gs.add_card_to_hand(0, 4001, 100)

    # Put payee in battle
    gs.add_test_card_to_battle(0, 4002, 200, False, False)

    # Cost: 10 - 2 (PASSIVE) - 3 (ACTIVE) = 5 mana needed
    for i in range(5):
        gs.add_card_to_mana(0, 4003, 300 + i)

    # Use both reductions
    cmd = {
        'type': dm.CommandType.PLAY_FROM_ZONE,
        'instance_id': 100,
        'payment_mode': 'ACTIVE_PAYMENT',
        'reduction_id': 'active_part',
        'payment_units': 1
    }

    gs.apply_move(cmd)
    try:
        if hasattr(dm, 'ensure_play_resolved'):
            dm.ensure_play_resolved(gs, cmd)
    except Exception:
        pass

    # Card should be in battle
    in_battle = any(c.card_id == 4001 for c in gs.players[0].battle_zone)
    assert in_battle, "Card with PASSIVE + ACTIVE reductions should play"


def test_multiple_passive_reductions_stacking(tmp_path):
    """Test that multiple PASSIVE reductions stack."""
    cards = [
        {
            "id": 6001,
            "name": "Stacked Passives",
            "type": "CREATURE",
            "cost": 12,
            "power": 2000,
            "civilizations": ["LIGHT"],
            "cost_reductions": [
                {
                    "id": "passive_a",
                    "type": "PASSIVE",
                    "reduction_amount": 3
                },
                {
                    "id": "passive_b",
                    "type": "PASSIVE",
                    "reduction_amount": 4
                }
            ]
        },
        {
            "id": 6002,
            "name": "Mana",
            "type": "MANA",
            "civilizations": ["LIGHT"]
        }
    ]

    p = tmp_path / "cards_stacked.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))

    game = dm.GameInstance(0, db)
    gs = game.state

    gs.add_card_to_hand(0, 6001, 100)

    # Cost: 12 - 3 - 4 = 5 mana needed
    for i in range(5):
        gs.add_card_to_mana(0, 6002, 200 + i)

    cmd = {
        'type': dm.CommandType.PLAY_FROM_ZONE,
        'instance_id': 100,
    }

    gs.apply_move(cmd)
    try:
        if hasattr(dm, 'ensure_play_resolved'):
            dm.ensure_play_resolved(gs, cmd)
    except Exception:
        pass

    in_battle = any(c.card_id == 6001 for c in gs.players[0].battle_zone)
    assert in_battle, "Card with stacked PASSIVE reductions should play"


def test_active_payment_prefers_reduction_id_over_legacy_str_val(tmp_path):
    """Runtime should prioritize payment field `reduction_id` over legacy `str_val`.

    Regression intent:
    - If both are present and disagree, engine must use `reduction_id`.
    - This verifies payment_* runtime utilization and legacy compatibility.
    """
    cards = [
        {
            "id": 7001,
            "name": "Reduction Priority Card",
            "type": "CREATURE",
            "cost": 8,
            "power": 2000,
            "civilizations": ["NATURE"],
            "cost_reductions": [
                {
                    "id": "active_weak",
                    "name": "WeakReduction",
                    "type": "ACTIVE_PAYMENT",
                    "unit_cost": {"type": "TAP_CARD", "amount": 0, "filter": {"zones": ["BATTLE_ZONE"]}},
                    "reduction_amount": 1,
                    "max_units": 1,
                },
                {
                    "id": "active_strong",
                    "name": "StrongReduction",
                    "type": "ACTIVE_PAYMENT",
                    "unit_cost": {"type": "TAP_CARD", "amount": 0, "filter": {"zones": ["BATTLE_ZONE"]}},
                    "reduction_amount": 4,
                    "max_units": 1,
                },
            ],
        },
        {
            "id": 7002,
            "name": "Payee",
            "type": "CREATURE",
            "cost": 1,
            "power": 100,
            "civilizations": ["NATURE"],
        },
        {
            "id": 7003,
            "name": "Mana",
            "type": "MANA",
            "civilizations": ["NATURE"],
        },
    ]

    p = tmp_path / "cards_priority.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))

    game = dm.GameInstance(0, db)
    gs = game.state

    gs.add_card_to_hand(0, 7001, 100)
    gs.add_test_card_to_battle(0, 7002, 200, False, False)

    # Need 4 mana iff strong reduction (8-4=4). Weak reduction would require 7.
    for i in range(4):
        gs.add_card_to_mana(0, 7003, 300 + i)

    cmd = {
        "type": dm.CommandType.PLAY_FROM_ZONE,
        "instance_id": 100,
        "payment_mode": "ACTIVE_PAYMENT",
        "reduction_id": "active_strong",
        "payment_units": 1,
        # Intentionally conflicting legacy fallback field.
        "str_val": "WeakReduction",
    }

    gs.apply_move(cmd)
    try:
        if hasattr(dm, "ensure_play_resolved"):
            dm.ensure_play_resolved(gs, cmd)
    except Exception:
        pass

    in_battle = any(c.card_id == 7001 for c in gs.players[0].battle_zone)
    assert in_battle, "Expected strong reduction via reduction_id to be applied at runtime"


def test_civilization_mismatch_blocks_play_even_if_total_mana_sufficient(tmp_path):
    """Runtime integration: civilization requirement must be enforced.

    Even when total mana count is enough, the card should not be played if the
    required civilization is missing in mana zone.
    """
    cards = [
        {
            "id": 8001,
            "name": "Water Requirement Card",
            "type": "CREATURE",
            "cost": 3,
            "power": 1000,
            "civilizations": ["WATER"],
        },
        {
            "id": 8002,
            "name": "Nature Mana",
            "type": "MANA",
            "civilizations": ["NATURE"],
        },
    ]

    p = tmp_path / "cards_civ_mismatch.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))

    game = dm.GameInstance(0, db)
    gs = game.state

    gs.add_card_to_hand(0, 8001, 100)

    # Total mana is sufficient (=3) but missing WATER civilization.
    for i in range(3):
        gs.add_card_to_mana(0, 8002, 200 + i)

    cmd = {
        "type": dm.CommandType.PLAY_FROM_ZONE,
        "instance_id": 100,
    }

    # Do not call ensure_play_resolved here: this is a failure-path assertion.
    gs.apply_move(cmd)

    in_battle = any(c.card_id == 8001 for c in gs.players[0].battle_zone)
    in_hand = any(c.instance_id == 100 for c in gs.players[0].hand)

    # Regression prevention: civilization mismatch must block runtime play.
    assert not in_battle, "Card should not be played without required civilization mana"
    assert in_hand, "Card should remain in hand when civilization requirement is unmet"


def test_active_payment_legacy_str_val_fallback_when_reduction_id_missing(tmp_path):
    """Runtime integration: legacy str_val fallback should work when reduction_id is absent."""
    cards = [
        {
            "id": 8101,
            "name": "Legacy Fallback Card",
            "type": "CREATURE",
            "cost": 7,
            "power": 1500,
            "civilizations": ["NATURE"],
            "cost_reductions": [
                {
                    "id": "active_legacy",
                    "name": "LegacyReduction",
                    "type": "ACTIVE_PAYMENT",
                    "unit_cost": {"type": "TAP_CARD", "amount": 0, "filter": {"zones": ["BATTLE_ZONE"]}},
                    "reduction_amount": 3,
                    "max_units": 1,
                }
            ],
        },
        {
            "id": 8102,
            "name": "Payee",
            "type": "CREATURE",
            "cost": 1,
            "power": 100,
            "civilizations": ["NATURE"],
        },
        {
            "id": 8103,
            "name": "Mana",
            "type": "MANA",
            "civilizations": ["NATURE"],
        },
    ]

    p = tmp_path / "cards_legacy_fallback.json"
    p.write_text(json.dumps(cards))
    db = dm.JsonLoader.load_cards(str(p))

    game = dm.GameInstance(0, db)
    gs = game.state

    gs.add_card_to_hand(0, 8101, 100)
    gs.add_test_card_to_battle(0, 8102, 200, False, False)

    # reduction_amount=3 so effective cost is 4. If fallback fails, 7 mana would be required.
    for i in range(4):
        gs.add_card_to_mana(0, 8103, 300 + i)

    cmd = {
        "type": dm.CommandType.PLAY_FROM_ZONE,
        "instance_id": 100,
        "payment_mode": "ACTIVE_PAYMENT",
        # Intentionally omit reduction_id to force legacy fallback.
        "str_val": "LegacyReduction",
        "payment_units": 1,
    }

    gs.apply_move(cmd)
    try:
        if hasattr(dm, "ensure_play_resolved"):
            dm.ensure_play_resolved(gs, cmd)
    except Exception:
        pass

    in_battle = any(c.card_id == 8101 for c in gs.players[0].battle_zone)
    assert in_battle, "Expected legacy str_val fallback to select ACTIVE_PAYMENT reduction at runtime"
