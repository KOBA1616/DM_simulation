#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a minimal simulation using the `standard_start` scenario from data/scenarios.json.
"""
import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    import dm_ai_module as dm
except Exception as e:
    print("dm_ai_module import failed:", e)
    raise


def load_standard():
    with open(project_root / "data" / "scenarios.json", "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    for s in scenarios:
        if s.get('name') == 'standard_start':
            return s.get('config', {})
    return {}


def ensure_list_length(lst, n, filler=1):
    while len(lst) < n:
        lst.append(filler)


def main():
    print("Run standard minimal: start")
    cfg = load_standard()
    print("Loaded scenario config:", cfg)

    # load native card db if possible
    card_db = None
    try:
        if hasattr(dm, 'JsonLoader'):
            card_db = dm.JsonLoader.load_cards("data/cards.json")
            print("Loaded native card_db via JsonLoader")
    except Exception as e:
        print("JsonLoader load failed:", e)

    if card_db is None:
        card_db = {}

    # create game instance
    try:
        gi = dm.GameInstance(0, card_db)
    except TypeError:
        gi = dm.GameInstance(0)

    state = gi.state
    state.setup_test_duel()

    # Prepare decks (40 cards id 1)
    deck = [1] * 40
    try:
        state.set_deck(0, deck)
        state.set_deck(1, deck)
    except Exception:
        # fallback: set lists directly
        state.players[0].deck = list(deck)
        state.players[1].deck = list(deck)

    # Apply scenario for each player
    # Hands
    for pid, prefix in ((0, 'my_'), (1, 'enemy_')):
        p = state.players[pid]
        hand_key = prefix + 'hand_cards'
        shields_key = prefix + ('shields' if pid==0 else 'shields')
        mana_key = prefix + 'mana_zone'
        battle_key = prefix + 'battle_zone'
        grave_key = prefix + 'grave_yard'

        # Hand: take from deck when possible so deck shrinks
        hand_cards = cfg.get(hand_key, [])
        try:
            p.hand.clear()
            for c in hand_cards:
                try:
                    # prefer drawing from deck
                    card = p.deck.pop()
                    p.hand.append(card)
                except Exception:
                    # fallback: use declared card id
                    p.hand.append(c)
        except Exception:
            p.hand = list(hand_cards)

        # Shields: take from deck when possible so deck shrinks
        shield_cards = cfg.get(shields_key, [])
        try:
            p.shield_zone.clear()
            for c in shield_cards:
                try:
                    card = p.deck.pop()
                    p.shield_zone.append(card)
                except Exception:
                    p.shield_zone.append(c)
        except Exception:
            p.shield_zone = list(shield_cards)

        # Mana: store as mana_zone list length
        try:
            p.mana_zone.clear()
            for c in cfg.get(mana_key, []):
                p.mana_zone.append(c)
        except Exception:
            p.mana_zone = list(cfg.get(mana_key, []))

        # Battle zone and grave
        try:
            p.battle_zone.clear()
            for c in cfg.get(battle_key, []):
                p.battle_zone.append(c)
        except Exception:
            p.battle_zone = list(cfg.get(battle_key, []))

        try:
            p.grave_yard.clear()
            for c in cfg.get(grave_key, []):
                p.grave_yard.append(c)
        except Exception:
            p.grave_yard = list(cfg.get(grave_key, []))

    # Log state
    print(f"P0 deck size: {len(state.players[0].deck)}, hand: {len(state.players[0].hand)}, shields: {len(state.players[0].shield_zone)}")
    print(f"P1 deck size: {len(state.players[1].deck)}, hand: {len(state.players[1].hand)}, shields: {len(state.players[1].shield_zone)}")

    # Try to invoke native start if available
    try:
        if hasattr(dm, 'PhaseManager') and hasattr(dm.PhaseManager, 'start_game'):
            dm.PhaseManager.start_game(state, card_db)
            print('Called native PhaseManager.start_game')
    except Exception as e:
        print('PhaseManager.start_game failed:', e)

    # Print final sample
    print('Final sample:')
    for pid in (0,1):
        p = state.players[pid]
        print(f'P{pid} hand preview:', list(p.hand)[:10])
        print(f'P{pid} shield preview:', list(p.shield_zone)[:10])

if __name__ == '__main__':
    main()
