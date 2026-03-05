# -*- coding: utf-8 -*-
"""
Headless runner for card id=1 behaviors.
Runs several scenarios and prints pipeline/log traces and hand/deck counts.
"""
from __future__ import annotations
import json
import os
import sys
from typing import Any, List

import sys
import pathlib

# Ensure repo root and native extension dir are on sys.path so dm_ai_module can be imported
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
native_dir = repo_root / "bin" / "Release"
if native_dir.exists():
    sys.path.insert(0, str(native_dir))

import dm_ai_module

CARDS_JSON = os.path.join(os.path.dirname(__file__), "..", "data", "cards.json")


def setup_game(card_id: int = 1, mana_cost: int = 2, seed: int = 42) -> Any:
    db = dm_ai_module.JsonLoader.load_cards(CARDS_JSON)
    game = dm_ai_module.GameInstance(seed, db)
    s = game.state
    s.set_deck(0, [card_id] * 40)
    s.set_deck(1, [1] * 40)
    dm_ai_module.PhaseManager.start_game(s, db)
    for i in range(mana_cost):
        s.add_card_to_mana(0, card_id, 9000 + i)
    # advance to MAIN (P0)
    for _ in range(30):
        ph = str(s.current_phase).upper()
        if "MAIN" in ph and s.active_player_id == 0:
            break
        legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        if legal:
            game.resolve_command(legal[0])
        else:
            dm_ai_module.PhaseManager.next_phase(s, db)
    return game, db


def play_and_resolve(game: Any, db: Any, select_choice: int | None) -> dict:
    s = game.state
    out = {"events": [], "hand_before": None, "hand_after": None, "deck_before": None, "deck_after": None}

    # find play
    legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
    play_cmd = next((c for c in legal if "PLAY" in str(getattr(c, "type", "")).upper()), None)
    if play_cmd is None:
        out["events"].append("no_play_cmd")
        return out

    out["hand_before"] = len(s.players[0].hand)
    out["deck_before"] = len(s.players[0].deck)

    game.resolve_command(play_cmd)
    out["events"].append("played_card")

    # find RESOLVE_EFFECT and execute
    legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
    resolve_cmd = next((c for c in legal if "RESOLVE" in str(getattr(c, "type", "")).upper()), None)
    if resolve_cmd is None:
        out["events"].append("no_resolve_cmd")
        return out

    game.resolve_command(resolve_cmd)
    out["events"].append("resolved_effect")

    # Now pipeline may request input (SELECT_NUMBER). Loop responding according to select_choice
    for step in range(10):
        legal = dm_ai_module.IntentGenerator.generate_legal_commands(s, db)
        if not legal:
            out["events"].append("no_more_commands")
            break
        # find select number
        sel = next((c for c in legal if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper()), None)
        if sel:
            # choose according to select_choice
            if select_choice is None:
                # pick first available
                choice = sel
            else:
                # find command with target_instance == select_choice if exists
                choice = next((c for c in legal if "SELECT_NUMBER" in str(getattr(c, "type", "")).upper() and getattr(c, "target_instance", -999) == select_choice), sel)
            out["events"].append(f"select_number_chosen={getattr(choice,'target_instance',None)}")
            game.resolve_command(choice)
            continue
        # SELECT_FROM_BUFFER
        sel_buf = next((c for c in legal if "SELECT_FROM_BUFFER" in str(getattr(c, "type", "")).upper()), None)
        if sel_buf:
            out["events"].append(f"select_from_buffer_instance={getattr(sel_buf,'instance_id',None)}")
            game.resolve_command(sel_buf)
            continue

        # otherwise resolve other commands
        # execute first
        game.resolve_command(legal[0])
        out["events"].append(f"executed_{str(getattr(legal[0],'type', ''))}")

    out["hand_after"] = len(s.players[0].hand)
    out["deck_after"] = len(s.players[0].deck)
    return out


if __name__ == "__main__":
    scenarios = [
        {"name": "select_zero", "seed": 100, "choice": 0},
        {"name": "select_max", "seed": 200, "choice": None},  # None -> first available (likely max)
        {"name": "select_one", "seed": 300, "choice": 1},
    ]

    results = {}
    for sc in scenarios:
        print("--- RUN", sc["name"], "seed", sc["seed"], "choice", sc["choice"])
        game, db = setup_game(card_id=1, mana_cost=2, seed=sc["seed"])
        res = play_and_resolve(game, db, sc["choice"])
        print(json.dumps(res, ensure_ascii=False, indent=2))
        results[sc["name"]] = res

    # summary
    print("=== SUMMARY ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))
