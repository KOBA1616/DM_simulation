import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dm_ai_module
from dm_ai_module import JsonLoader
from dm_toolkit import commands


def test_generate_play_candidates_present():
    # Setup game instance and card DB
    # Ensure internal DB is loaded for CommandSystem
    try:
        JsonLoader.load_cards("data/cards.json")
    except Exception:
        pass

    try:
        gi = dm_ai_module.GameInstance(0)
    except TypeError:
        gi = dm_ai_module.GameInstance()
    state = gi.state
    # Some builds expose initialize(); call if present
    if hasattr(state, 'initialize') and callable(state.initialize):
        try:
            state.initialize()
        except Exception:
            pass

    # Minimal card DB: id->dict with cost
    # Note: This python DB is passed to generate_legal_commands, but the native engine uses its own.
    card_db = {
        1: {"id": 1, "name": "Cheap Creature", "cost": 1, "type": "CREATURE"},
        99: {"id": 99, "name": "Expensive", "cost": 99, "type": "CREATURE"}
    }

    # Place cheap card in hand for active player 0
    state.active_player_id = 0
    # Set MAIN phase robustly (native builds may expect a Phase enum)
    try:
        state.current_phase = dm_ai_module.Phase.MAIN
    except Exception:
        state.current_phase = 3  # fallback integer

    # Add the cheap card to hand (handle native overloads)
    try:
        state.add_card_to_hand(0, 1, 100) # player, card, instance
    except TypeError:
        state.add_card_to_hand(0, 1)

    # Give one untapped mana (handle native overloads)
    # Using ID 1 (Water) which matches the card we are trying to play (ID 1 is Water creature)
    try:
        state.add_card_to_mana(0, 1, 101)
    except TypeError:
        state.add_card_to_mana(0, 1)

    # Ensure mana cards are untapped if the attribute exists
    try:
        for m in state.players[0].mana_zone:
            m.is_tapped = False
    except Exception:
        pass

    # Call generate_legal_commands
    cmds = commands.generate_legal_commands(state, card_db)

    # Expect at least one play-like command
    found_play = False
    for c in cmds:
        try:
            d = c.to_dict()
            # Check unified hint or PLAY_FROM_ZONE
            if d.get('unified_type') == 'PLAY' or d.get('type') == dm_ai_module.CommandType.PLAY_FROM_ZONE:
                found_play = True
                break
        except Exception:
            continue

    assert found_play, f"No play candidate found in cmds: {[c.to_string() for c in cmds]}"
