#!/usr/bin/env python3
"""
Helper script to generate JSON test scenarios.
Usage:
    python create_scenario.py --name "my_test" --player 0 --hand 1 2 3 --mana 4 5 --action MANA_CHARGE --card-idx 0
"""

import argparse
import json
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Create a test scenario JSON file.")
    parser.add_argument("--name", type=str, required=True, help="Name of the scenario file (without extension)")
    parser.add_argument("--player", type=int, default=0, help="Turn player index (0 or 1)")

    # Player 0 Setup
    parser.add_argument("--hand", type=int, nargs='+', default=[], help="Card IDs in Player 0 hand")
    parser.add_argument("--mana", type=int, nargs='+', default=[], help="Card IDs in Player 0 mana zone")
    parser.add_argument("--battle", type=int, nargs='+', default=[], help="Card IDs in Player 0 battle zone")
    parser.add_argument("--shield", type=int, nargs='+', default=[], help="Card IDs in Player 0 shield zone")

    # Player 1 Setup (simplified for CLI, can edit file later)
    parser.add_argument("--opp-hand", type=int, nargs='+', default=[], help="Card IDs in Player 1 hand")
    parser.add_argument("--opp-mana", type=int, nargs='+', default=[], help="Card IDs in Player 1 mana zone")
    parser.add_argument("--opp-battle", type=int, nargs='+', default=[], help="Card IDs in Player 1 battle zone")
    parser.add_argument("--opp-shield", type=int, nargs='+', default=[], help="Card IDs in Player 1 shield zone")

    # Action
    parser.add_argument("--action", type=str, required=True, choices=["MANA_CHARGE"], help="Action type")
    parser.add_argument("--action-player", type=int, default=0, help="Player performing action")
    parser.add_argument("--card-idx", type=int, default=0, help="Card index for action")

    # Expected (Simple counts)
    parser.add_argument("--expect-mana", type=int, help="Expected mana count for player 0")
    parser.add_argument("--expect-hand", type=int, help="Expected hand count for player 0")

    args = parser.parse_args()

    scenario = {
        "description": f"Generated scenario: {args.name}",
        "setup": {
            "player_turn": args.player,
            "players": [
                {
                    "hand": args.hand,
                    "mana_zone": args.mana,
                    "battle_zone": args.battle,
                    "shield_zone": args.shield
                },
                {
                    "hand": args.opp_hand,
                    "mana_zone": args.opp_mana,
                    "battle_zone": args.opp_battle,
                    "shield_zone": args.opp_shield
                }
            ]
        },
        "action": {
            "type": args.action,
            "player_index": args.action_player,
            "card_index": args.card_idx
        },
        "expected": {}
    }

    if args.expect_mana is not None:
        scenario["expected"]["mana_zone_count_p0"] = args.expect_mana
    if args.expect_hand is not None:
        scenario["expected"]["hand_count_p0"] = args.expect_hand

    # Output file
    output_dir = os.path.join(os.path.dirname(__file__), "scenarios")
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{args.name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(scenario, f, indent=2)

    print(f"Scenario created at: {filepath}")

if __name__ == "__main__":
    main()
