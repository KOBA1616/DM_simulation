import sys
import os
import pytest

# Ensure module can be imported
sys.path.append(os.path.join(os.getcwd(), 'build'))

try:
    import dm_ai_module
except ImportError:
    # Try alternate path if build is not in root/build (e.g. if run from within build or python dir)
    sys.path.append(os.path.join(os.getcwd(), '../build'))
    import dm_ai_module

def test_token_encoding_basic():
    # Setup state
    state = dm_ai_module.GameState(100) # 100 cards
    # Add some cards
    # ID=1, Instance=0, Hand
    state.add_card_to_hand(0, 1, 0)
    # ID=2, Instance=1, Mana
    state.add_card_to_mana(0, 2, 1)

    # Encode
    tokens = dm_ai_module.TokenConverter.encode_state(state, 0, 128)

    assert len(tokens) == 128
    assert tokens[0] == 1 # CLS
    # Check for markers
    assert 10 in tokens # HAND_SELF
    assert 11 in tokens # MANA_SELF
    assert 14 in tokens # GRAVE_SELF (New)
    assert 15 in tokens # DECK_SELF (New)
    assert 24 in tokens # GRAVE_OPP (New)
    assert 25 in tokens # DECK_OPP (New)

    # Check Phase (Context)
    # 100 is BASE_CONTEXT_MARKER.
    # Turn is pushed, then Phase.
    # Phase is now offset by BASE_PHASE_MARKER (80).
    # tokens[1] is 100 (Context Start)
    # tokens[2] is Turn (1)
    # tokens[3] is Phase
    assert tokens[1] == 100
    assert tokens[2] == 1
    # Phase default is START_OF_TURN (0) -> 80 + 0 = 80
    # Phase can be checked if it is >= 80
    assert tokens[3] >= 80

    # ID=1 is 1001 (BASE=1000)
    assert 1001 in tokens
    # ID=2 is 1002
    assert 1002 in tokens

def test_token_command_history():
    state = dm_ai_module.GameState(100)
    # We can't easily push commands from python binding unless exposed.
    # Assuming command_history is populated by engine actions.
    # For now, just check it doesn't crash on empty history.
    tokens = dm_ai_module.TokenConverter.encode_state(state, 0, 64)
    assert 200 not in tokens # No commands yet (checking specific CMD token if we knew it)
    # 2 = SEP
    assert 2 in tokens
