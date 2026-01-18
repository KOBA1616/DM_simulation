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

    # Use batch play to collect results deterministically from C++ runner
    total = args.games
    # Run half with challenger as P1, half with challenger as P2 by swapping argument order
    n1 = total // 2
    n2 = total - n1

    deck_a = list(deck)
    deck_b = list(deck)

    wins = 0
    try:
        if n1 > 0:
            res1 = eco.runner.play_deck_matchup(deck_a, deck_b, n1, 4)
            # Challenger is deck_a (P1) in this batch
            wins += res1.count(1)

        if n2 > 0:
            # Swap order so challenger is P2 in this batch
            res2 = eco.runner.play_deck_matchup(deck_b, deck_a, n2, 4)
            # Challenger is deck_a passed as second arg -> count P2 wins
            wins += res2.count(2)
    except Exception:
        wr, score = eco.evaluate_deck(deck, "self", num_games=args.games)
        print(f"Fallback evaluate_deck result: {wr*100:.1f}%, Smart Score: {score:.2f}")
        return

    wr = wins / total if total > 0 else 0.0
    # Compute a quick smart score via evaluate_deck's stats path for reporting
    _, score = eco.evaluate_deck(deck, "self", num_games=2)
    print(f"Identical-deck match win-rate: {wr*100:.1f}% ({wins}/{total}), Smart Score: {score:.2f}")

    # Restore print
    if args.quiet:
        builtins.print = orig_print

if __name__ == '__main__':
    main()
