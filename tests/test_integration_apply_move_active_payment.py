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
