#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a minimal headless simulation to verify game progression.
This script uses EvolutionEcosystem.collect_smart_stats to run a few games
between identical decks and prints aggregated stats.
"""
import os
import sys
import random
import argparse
import time
import builtins

# Ensure repository root and python/ are on sys.path so dm_toolkit imports resolve
sys.path.insert(0, os.getcwd())
if os.path.isdir("python"):
    sys.path.insert(0, os.path.abspath("python"))

from dm_toolkit.training.evolution_ecosystem import EvolutionEcosystem

def main():
    parser = argparse.ArgumentParser(description="Minimal headless simulation runner")
    parser.add_argument("--cards", default=os.path.join("data", "cards.json"))
    parser.add_argument("--meta", default=os.path.join("data", "meta_decks.json"))
    parser.add_argument("--games", type=int, default=100, help="Number of games for identical-deck matchup")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress EngineCompat noisy prints")
    args = parser.parse_args()

    cards_path = args.cards
    meta_path = args.meta

    eco = EvolutionEcosystem(cards_path, meta_path)

    # Optionally set deterministic seeds
    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np
            np.random.seed(args.seed)
        except Exception:
            pass
        try:
            import torch
            torch.manual_seed(args.seed)
        except Exception:
            pass

    # Create a random deck of 40 card ids from card_db
    valid_ids = list(eco.card_db.keys())
    if not valid_ids:
        print("No cards found in card DB.")
        return

    deck = [random.choice(valid_ids) for _ in range(40)]

    # Quiet mode: filter noisy EngineCompat / execute warnings by monkeypatching print
    orig_print = builtins.print
    def filtered_print(*args, **kwargs):
        try:
            if args and isinstance(args[0], str) and (args[0].startswith('EngineCompat') or 'ExecuteCommand could not execute' in args[0] or args[0].startswith('Warning: ExecuteCommand')):
                return
        except Exception:
            pass
        return orig_print(*args, **kwargs)

    if args.quiet:
        builtins.print = filtered_print

    # If seed provided, make time.time deterministic so collect_smart_stats uses stable seeds
    orig_time = time.time
    if args.seed is not None:
        time.time = lambda: float(args.seed)

    print(f"Running minimal simulation: same deck vs same deck (2 games for stats)...")
    stats = eco.collect_smart_stats(deck, deck, num_games=2)

    # Restore time/time behavior for runner if needed
    if args.seed is not None:
        time.time = orig_time

    # Print a small summary
    played = sorted(((cid, v['play'], v['resource']) for cid, v in stats.items()), key=lambda x: -x[1])
    print("Top played cards (cid, plays, resource):")
    for cid, plays, resource in played[:10]:
        print(f"  {cid}: plays={plays}, resource={resource}")

    print("Simulation complete.")

    # Evaluate same-deck matchup using ParallelRunner (C++ powered)
    print(f"\nRunning identical-deck matchup ({args.games} games) to measure win-rate (seed={args.seed})...")
    # Ensure meta contains at least one deck so evaluate_deck uses real runner path
    eco.meta_decks = [{"name": "self", "cards": deck}]

    # evaluate_deck may sample meta; instead run a deterministic alternating-order self-play
    # so that challenger is P1 in even games and P2 in odd games (fairness)
    wins = 0
    total = args.games
    for i in range(total):
        # Always call matchup with same deck pair; interpret result based on index parity
        try:
            res = eco.runner.play_deck_matchup(deck, deck, 1, 4)
        except Exception:
            # Fallback to evaluate_deck if runner fails
            wr, score = eco.evaluate_deck(deck, "self", num_games=args.games)
            print(f"Fallback evaluate_deck result: {wr*100:.1f}%, Smart Score: {score:.2f}")
            break

        if not res:
            continue

        winner = res[0]
        # If i is even, challenger acted as P1 -> win if winner == 1
        if i % 2 == 0:
            if winner == 1:
                wins += 1
        else:
            # odd: challenger considered as P2 -> win if winner == 2
            if winner == 2:
                wins += 1

    wr = wins / total if total > 0 else 0.0
    # Compute a quick smart score via evaluate_deck's stats path for reporting
    _, score = eco.evaluate_deck(deck, "self", num_games=2)
    print(f"Identical-deck match win-rate: {wr*100:.1f}% ({wins}/{total}), Smart Score: {score:.2f}")

    # Restore print
    if args.quiet:
        builtins.print = orig_print

if __name__ == '__main__':
    main()
