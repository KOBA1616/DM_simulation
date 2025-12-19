import sys
import os

# Add bin/ to path to find dm_ai_module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build'))
try:
    import dm_ai_module
    print(f"Loaded module from: {dm_ai_module.__file__}")
except ImportError:
    print("Could not import dm_ai_module. Ensure the module is built.")
    sys.exit(1)

def test_tokenization_basic():
    state = dm_ai_module.GameState(100)

    # Setup some cards
    # P1 (Self) Hand: Card 1
    state.add_card_to_hand(0, 1, 0)

    # P2 (Opp) Mana: Card 2
    state.add_card_to_mana(1, 2, 1)

    # Run Tokenization from P0 perspective
    tokens = dm_ai_module.TokenConverter.encode_state(state, 0, 512)

    print(f"Tokens P0: {tokens}")

    # Check Self Hand (10) has Card 1 (1001)
    assert 10 in tokens, "Missing MARKER_HAND_SELF"
    assert 1001 in tokens, "Missing Card 1 in Self Hand"

    # Check Opp Mana (21) has Card 2 (1002)
    assert 21 in tokens, "Missing MARKER_MANA_OPP"
    assert 1002 in tokens, "Missing Card 2 in Opp Mana"

    # Run Tokenization from P1 perspective (Swap)
    tokens_p1 = dm_ai_module.TokenConverter.encode_state(state, 1, 512)
    print(f"Tokens P1: {tokens_p1}")

    # Now P1 is Self.
    # Card 1 is in P0 Hand. P1 sees P0 as Opp. P0 Hand is Hidden.
    # So P0 Hand (Opp Hand) -> Zone 20. Content -> UNK (3).
    assert 20 in tokens_p1, "Missing MARKER_HAND_OPP (P1 view)"
    assert 3 in tokens_p1, "Missing UNK token in Hidden Hand"

    # Card 2 is in P1 Mana. P1 sees P1 as Self. P1 Mana -> Zone 11.
    assert 11 in tokens_p1, "Missing MARKER_MANA_SELF (P1 view)"
    assert 1002 in tokens_p1, "Missing Card 2 in Self Mana"

if __name__ == "__main__":
    test_tokenization_basic()
    print("All tests passed!")
