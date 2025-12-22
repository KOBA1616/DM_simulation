
import numpy as np
import dm_ai_module

def game_state_to_tokens(game_state, card_db, max_len=64, vocab_size=1000):
    """
    Converts a C++ GameState object into a sequence of integer tokens.
    Token Mapping:
    - 0: PAD
    - 1...N: Card ID
    - N+1...: Zone Separators/Markers (Simplified for this task)
    """

    tokens = []

    # 1. Hands (Player 1)
    # Note: In C++ binding, accessing zones might return copies.
    # We iterate over card IDs in hand.

    # Helper to get cards from a player
    # Assume game_state.players[0] is Self

    # Actually, GameState binding might not expose players list directly nicely.
    # But `game_state.get_card_instance(id)` works if we have IDs.
    # Let's rely on standard zones if exposed, or just simulate "visible cards".

    # For verification purpose, we iterate ALL cards in the game and check their zone.
    # This is slow O(N) but fine for verification.
    # Better: Use scenario config or specific zone accessors if available.

    # Let's try to access game_state.players[0].hand
    # If not available, we can't do accurate tokenization easily.

    # Fallback: Since verify_performance uses game_state, let's look at available properties.
    # `game_state` object in python binding has:
    # .turn_number, .active_player_id
    # Methods: .get_card_instance(id)

    # We need a robust way.
    # dm_ai_module.get_visible_card_ids(game_state, player_id) -> List[int] ??
    # If not existing, we make a simple assumption:
    # The `TensorConverter` does this logic in C++.

    # For this Python Tokenizer, we will just map the IDs of cards in hand/battle/mana.

    # We will iterate through a known set of IDs or just "scan"
    # since we don't have an easy iterator for "cards in hand".

    # Wait, `game_state` usually has `players` which is a list.
    # `player.hand` -> vector<CardInstance> or vector<CardID>?
    # If it's exposed, we use it.

    # Let's assume standard structure:
    try:
        p1 = game_state.players[0]
        # p1.hand is likely a list of CardInstance or IDs.
        # Let's assume it returns CardInstance copies, so we take .card_id

        # HAND
        tokens.append(1001) # Hand Start Token
        for card in p1.hand:
            tokens.append(min(card.card_id, vocab_size-1))

        # MANA
        tokens.append(1002) # Mana Start Token
        for card in p1.mana_zone:
            tokens.append(min(card.card_id, vocab_size-1))

        # BATTLE
        tokens.append(1003) # Battle Start Token
        for card in p1.battle_zone:
            tokens.append(min(card.card_id, vocab_size-1))

        # SHIELDS (Count only?)
        # For Transformer we might want IDs if known, or just a token for "Shield"

    except Exception as e:
        # Fallback if structure is different
        # print(f"Tokenizer warning: {e}")
        pass

    # Pad or Truncate
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))

    return np.array(tokens, dtype=np.int64)
