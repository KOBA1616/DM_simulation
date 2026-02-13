#!/usr/bin/env python3
"""Emit examples where PLAY and ATTACK appear and map to canonical indices.

Creates two GameState variants:
- a 'playable' state: active player has mana and a low-cost card in hand (Phase.MAIN)
- an 'attack' state: active player has a creature in battle and Phase.ATTACK

Prints normalized command dicts and their `CommandEncoder.command_to_index` values.
"""
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dm_toolkit.engine.compat import EngineCompat
from dm_toolkit import commands_v2 as commands
from dm_toolkit.training.command_compat import generate_legal_commands, command_to_index, normalize_to_command
try:
    import dm_ai_module
except Exception:
    # fallback to toolkit shim if available
    from dm_toolkit import dm_ai_module


def dump_mapped_commands(state, card_db, note: str):
    # Prefer command-first generator via compatibility helper
    cmds = generate_legal_commands(state, card_db, strict=False) or []

    out = []
    for w in cmds:
        try:
            d = normalize_to_command(w)
        except Exception:
            d = {"_repr": repr(w)}
        idx = command_to_index(w)
        out.append({'cmd': d, 'index': idx})

    print(json.dumps({'note': note, 'players': [len(getattr(p, 'hand', [])) for p in state.players], 'phase': str(getattr(state, 'current_phase', None)), 'mapped': out}, ensure_ascii=False, indent=2))


def make_playable_state():
    s = dm_ai_module.GameState()
    # active player 0
    s.active_player_id = 0
    # Give 1 untapped mana
    s.add_card_to_mana(0, card_id=1001, count=1)
    # Put a cheap card in hand with id 42
    s.add_card_to_hand(0, card_id=42)
    # Ensure we're in MAIN phase
    try:
        s.current_phase = dm_ai_module.Phase.MAIN
    except Exception:
        s.current_phase = 3
    return s


def make_attack_state():
    s = dm_ai_module.GameState()
    s.active_player_id = 0
    # Put a creature on battle zone (untapped, not sick)
    s.add_test_card_to_battle(0, card_id=2001, instance_id=7777, tapped=False, sick=False)
    # Ensure we're in ATTACK phase
    try:
        s.current_phase = dm_ai_module.Phase.ATTACK
    except Exception:
        s.current_phase = 4
    return s


def main():
    # Simple card_db mapping for cost lookup
    card_db = {42: {'cost': 1}, 1001: {'cost': 0}, 2001: {'cost': 0}}

    play_state = make_playable_state()
    dump_mapped_commands(play_state, card_db, 'playable_state')

    attack_state = make_attack_state()
    dump_mapped_commands(attack_state, card_db, 'attack_state')


if __name__ == '__main__':
    main()
