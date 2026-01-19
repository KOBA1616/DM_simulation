#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a single self-play game with verbose trace for inspection.
"""
import sys
import random
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import dm_ai_module as dm


def trace_one_game(seed: int | None = None):
    if seed is not None:
        random.seed(seed)

    card_db = dm.JsonLoader.load_cards("data/cards.json")
    gi = dm.GameInstance(0, card_db)
    deck = [1] * 40
    gi.state.set_deck(0, deck[:])
    gi.state.set_deck(1, deck[:])
    gi.start_game()

    # Randomize starting player for fairness (same logic as self_play)
    gi.state.active_player_id = random.choice([0, 1])

    print(f"Start: turn={gi.state.turn_number}, active={gi.state.active_player_id}")
    print(f"Shields start: P0={len(gi.state.players[0].shield_zone)} P1={len(gi.state.players[1].shield_zone)}")

    turn = 0
    # We'll simulate until winner or safety cap
    while gi.state.winner == dm.GameResult.NONE and turn < 100:
        turn += 1
        ap = gi.state.active_player_id
        target = 1 - ap

        print(f"\n[Turn {turn}] Active: P{ap} -> attack P{target}")
        before = len(getattr(gi.state.players[target], 'shield_zone', []))
        if before > 0:
            gi.state.players[target].shield_zone.pop()
            after = len(getattr(gi.state.players[target], 'shield_zone', []))
            print(f"  BROKE SHIELD: P{target} shields {before} -> {after}")
        else:
            # No shields -> attacker wins
            gi.state.winner = dm.GameResult.P1_WIN if ap == 0 else dm.GameResult.P2_WIN
            print(f"  NO SHIELDS -> P{ap} wins (set winner enum={gi.state.winner})")
            break

        # Alternate active player (simple turn mechanics used in self_play)
        gi.state.active_player_id = target

        # Optional: advance turn_number if both players acted (not strictly needed here)
        if turn % 2 == 0:
            gi.state.turn_number += 1

    print(f"\nGame end: winner={gi.state.winner} (int={int(gi.state.winner)}) after {turn} turns")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    trace_one_game(args.seed)
