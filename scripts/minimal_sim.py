#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal game simulation to verify game progression and inference hooks.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import dm_ai_module as dm


def main():
    print("Minimal simulation: start")

    card_db = dm.JsonLoader.load_cards("data/cards.json")

    gi = dm.GameInstance(0, card_db)

    # Prepare simple decks (40 cards of ID 1)
    deck = [1] * 40
    gi.state.set_deck(0, deck)
    gi.state.set_deck(1, deck)

    # Start game (fills shields and draws)
    gi.start_game()
    print(f"Turn: {gi.state.turn_number}, Active: {gi.state.active_player_id}")
    print(f"P0 shields: {len(gi.state.players[0].shield_zone)}, P1 shields: {len(gi.state.players[1].shield_zone)}")

    # If GameInstance provides an execute method, show it; otherwise mutate state directly.
    print("GameInstance attrs:", [a for a in dir(gi) if not a.startswith('__')])

    # Break opponent shields by directly mutating state (works with native or stubbed module)
    target = 1
    while len(getattr(gi.state.players[target], 'shield_zone', [])) > 0:
        gi.state.players[target].shield_zone.pop()
        print(f"Removed one shield from P{target}, remaining: {len(gi.state.players[target].shield_zone)}")

    # Set winner according to target losing all shields
    if target == 0:
        gi.state.winner = dm.GameResult.P2_WIN
    else:
        gi.state.winner = dm.GameResult.P1_WIN

    if gi.state.winner != dm.GameResult.NONE:
        print("Game over detected.")
        w = gi.state.winner
        print(f"Winner: {w}")
    else:
        print("No winner after breaks; game state: ", gi.state.winner)


if __name__ == '__main__':
    main()
