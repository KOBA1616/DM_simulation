import numpy as np

# Token IDs
PAD_TOKEN = 0
START_TOKEN = 1
END_TOKEN = 2

# Zone Markers
HAND_SELF = 10
MANA_SELF = 11
BATTLE_SELF = 12
SHIELD_SELF = 13
GRAVE_SELF = 14

HAND_OPP = 20
MANA_OPP = 21
BATTLE_OPP = 22
SHIELD_OPP = 23
GRAVE_OPP = 24

TAPPED_MARKER = 30
# Card ID Offset to avoid collision with markers
# Assuming markers < 100.
CARD_OFFSET = 100

class StateTokenizer:
    @staticmethod
    def tokenize(state, player_id: int, max_seq_len: int = 200, full_info: bool = True) -> np.ndarray:
        """
        Converts GameState into a sequence of tokens.

        Args:
            state: GameState object
            player_id: ID of the perspective player (0 or 1)
            max_seq_len: Maximum sequence length
            full_info: If True, includes hidden zones (Opp Hand, Shields).
                       If False, mimics observed state.
        """
        tokens = [START_TOKEN]
        opp_id = 1 - player_id

        # Helper to add card tokens
        def add_zone_tokens(cards, zone_marker, is_battle=False, hide_ids=False):
            tokens.append(zone_marker)
            for card in cards:
                if is_battle and card.is_tapped:
                     tokens.append(TAPPED_MARKER)

                if hide_ids:
                    # Placeholder for hidden card (could be a generic 'CARD_BACK' token)
                    # For now, let's just skip adding ID or add a dummy ID like 99
                    tokens.append(99)
                else:
                    # Ensure card_id is valid
                    cid = card.card_id
                    tokens.append(cid + CARD_OFFSET)

        # Self Zones (Always Fully Visible)
        # Access via state.players[pid]
        p_self = state.players[player_id]
        p_opp = state.players[opp_id]

        add_zone_tokens(p_self.hand, HAND_SELF)
        add_zone_tokens(p_self.mana_zone, MANA_SELF)
        add_zone_tokens(p_self.battle_zone, BATTLE_SELF, is_battle=True)
        add_zone_tokens(p_self.shield_zone, SHIELD_SELF)
        add_zone_tokens(p_self.graveyard, GRAVE_SELF)

        # Opponent Zones
        # Visible
        add_zone_tokens(p_opp.mana_zone, MANA_OPP)
        add_zone_tokens(p_opp.battle_zone, BATTLE_OPP, is_battle=True)
        add_zone_tokens(p_opp.graveyard, GRAVE_OPP)

        # Hidden or Semi-Hidden
        if full_info:
            add_zone_tokens(p_opp.hand, HAND_OPP)
            add_zone_tokens(p_opp.shield_zone, SHIELD_OPP)
        else:
            # For masked state, we might see Hand Size but not IDs.
            # Implementation choice: Tokenize 'Hidden Card' x Count
            add_zone_tokens(p_opp.hand, HAND_OPP, hide_ids=True)
            add_zone_tokens(p_opp.shield_zone, SHIELD_OPP, hide_ids=True)

        tokens.append(END_TOKEN)

        # Pad or Truncate
        if len(tokens) > max_seq_len:
            tokens = tokens[:max_seq_len]
        else:
            tokens.extend([PAD_TOKEN] * (max_seq_len - len(tokens)))

        return np.array(tokens, dtype=np.int64)
