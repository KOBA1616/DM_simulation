#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-play simulation between identical decks to measure win rate.
Simple symmetric policy: active player attacks opponent each turn.
"""
import sys
from pathlib import Path
import random

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import dm_ai_module as dm


def play_once(seed: int | None = None) -> int:
    print("WARNING: This Python-based game loop is deprecated. Please use `dm-cli sim` which uses the C++ engine.")
    return 0
    # if seed is not None:
    #     random.seed(seed)

    # card_db = dm.JsonLoader.load_cards("data/cards.json")
    # gi = dm.GameInstance(0, card_db)
    # # Prepare identical decks
    # deck = [1] * 40
    # gi.state.set_deck(0, deck[:])
    # gi.state.set_deck(1, deck[:])
    # gi.start_game()

    # # Optionally randomize starting player for fairness
    # # If attribute exists, we can set it; default keep what engine provides
    # if hasattr(gi.state, 'active_player_id'):
    #     # 50 random starting player
    #     gi.state.active_player_id = random.choice([0, 1])

    # # Simple loop: active player attacks opponent (break shield) until winner.
    # # Use available API: mutate shield_zone directly if engine doesn't expose execute.
    # while gi.state.winner == dm.GameResult.NONE:
    #     ap = gi.state.active_player_id
    #     target = 1 - ap

    #     # If shields present, remove one
    #     if len(getattr(gi.state.players[target], 'shield_zone', [])) > 0:
    #         gi.state.players[target].shield_zone.pop()
    #     else:
    #         # set winner (target has no shields -> attacker wins)
    #         gi.state.winner = dm.GameResult.P1_WIN if ap == 0 else dm.GameResult.P2_WIN
    #         break

    #     # Advance turn manually if PhaseManager not updating: alternate active player
    #     gi.state.active_player_id = target

    #     # Safety cap to avoid infinite loop
    #     if gi.state.turn_number > 1000:
    #         break

    # return int(gi.state.winner)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    counts = {}
    for i in range(args.games):
        seed = None if args.seed is None else args.seed + i
        res = play_once(seed)
        counts[res] = counts.get(res, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"Played {i+1}/{args.games} games...")

    # Map numeric results to labels using GameResult constants
    def label_for(val):
        if val == int(dm.GameResult.NONE):
            return 'NONE'
        if val == int(dm.GameResult.P1_WIN):
            return 'P1_WIN'
        if val == int(dm.GameResult.P2_WIN):
            return 'P2_WIN'
        if val == int(dm.GameResult.DRAW):
            return 'DRAW'
        return str(val)

    total = sum(counts.values())
    print('\nSelf-play results:')
    for k, v in sorted(counts.items()):
        print(f"  {label_for(k)}: {v}")
    print(f"  Total: {total}")



if __name__ == '__main__':
    main()
