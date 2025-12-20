#include "token_converter.hpp"
#include <algorithm>

namespace dm::ai::encoders {

    std::vector<int> TokenConverter::encode_state(const dm::core::GameState& state, int perspective, int max_len) {
        std::vector<int> tokens;
        tokens.reserve(512);

        if (state.players.size() < 2) {
             // Should not happen in standard game, but robust return
             return tokens;
        }

        // 1. CLS Token
        tokens.push_back(TOKEN_CLS);

        // 2. Game Metadata (Turn, Mana)
        tokens.push_back(BASE_CONTEXT_MARKER + 0); // Context Start
        // Turn
        tokens.push_back(state.turn_number);
        // Phase (New) - Offset to avoid collision with 0 (PAD)
        tokens.push_back(BASE_PHASE_MARKER + static_cast<int>(state.current_phase));
        // My Mana
        tokens.push_back(state.players[perspective].mana_zone.size());
        // Opp Mana
        tokens.push_back(state.players[1 - perspective].mana_zone.size());

        // 3. Zones
        const auto& self = state.players[perspective];
        const auto& opp = state.players[1 - perspective];

        // Self Battle
        append_zone(tokens, self.battle_zone, MARKER_BATTLE_SELF, true);
        // Opp Battle
        append_zone(tokens, opp.battle_zone, MARKER_BATTLE_OPP, true);

        // Self Mana
        append_zone(tokens, self.mana_zone, MARKER_MANA_SELF, true);
        // Opp Mana
        append_zone(tokens, opp.mana_zone, MARKER_MANA_OPP, true); // Usually visible

        // Self Hand
        append_zone(tokens, self.hand, MARKER_HAND_SELF, true);
        // Opp Hand (Masked count)
        tokens.push_back(MARKER_HAND_OPP);
        for(size_t i=0; i<opp.hand.size(); ++i) {
            tokens.push_back(TOKEN_UNK); // Or just count
        }

        // Shields (Masked)
        tokens.push_back(MARKER_SHIELD_SELF);
        for(size_t i=0; i<self.shield_zone.size(); ++i) tokens.push_back(TOKEN_UNK);

        tokens.push_back(MARKER_SHIELD_OPP);
        for(size_t i=0; i<opp.shield_zone.size(); ++i) tokens.push_back(TOKEN_UNK);

        // Graveyard (New)
        append_zone(tokens, self.graveyard, MARKER_GRAVE_SELF, true);
        append_zone(tokens, opp.graveyard, MARKER_GRAVE_OPP, true); // Public zone

        // Deck (New)
        // Self Deck (Masked usually, but owner might know top? For now masked)
        tokens.push_back(MARKER_DECK_SELF);
        for(size_t i=0; i<self.deck.size(); ++i) tokens.push_back(TOKEN_UNK);

        // Opp Deck (Masked)
        tokens.push_back(MARKER_DECK_OPP);
        for(size_t i=0; i<opp.deck.size(); ++i) tokens.push_back(TOKEN_UNK);


        // 4. Command History (Last N commands)
        tokens.push_back(TOKEN_SEP);
        append_command_history(tokens, state, 10);

        // Padding
        if (max_len > 0) {
            if (tokens.size() > max_len) {
                tokens.resize(max_len);
            } else {
                while(tokens.size() < max_len) {
                    tokens.push_back(TOKEN_PAD);
                }
            }
        }

        return tokens;
    }

    void TokenConverter::append_card(std::vector<int>& tokens, const dm::core::CardInstance& card, bool visible) {
        if (!visible) {
            tokens.push_back(TOKEN_UNK);
            return;
        }

        // Card ID base
        tokens.push_back(BASE_CARD_ID + card.card_id);

        // Status Flags
        if (card.is_tapped) tokens.push_back(STATE_TAPPED);
        if (card.summoning_sickness) tokens.push_back(STATE_SICK);
        // Face down is usually covered by 'visible' arg, but if it's visible to owner but face down (e.g. shield check?)
        // Standard face down (shields, mana) logic usually handled by caller.
    }

    void TokenConverter::append_zone(std::vector<int>& tokens, const std::vector<dm::core::CardInstance>& zone, int zone_token, bool visible) {
        tokens.push_back(zone_token);
        for (const auto& card : zone) {
            append_card(tokens, card, visible);
        }
    }

    void TokenConverter::append_command_history(std::vector<int>& tokens, const dm::core::GameState& state, int limit) {
        // Collect last N commands in chronological order
        // state.command_history is append-only, so end() is newest.

        int n = state.command_history.size();
        int start_idx = std::max(0, n - limit);

        for (int i = start_idx; i < n; ++i) {
            const auto& cmd = state.command_history[i];

            // Map command type to token
            int cmd_type_token = BASE_COMMAND_MARKER + (int)cmd->get_type();
            tokens.push_back(cmd_type_token);

            // TODO: Extract card_instance_id or other details if available in base GameCommand
            // Currently GameCommand base class might not expose card_id directly without casting.
            // For now, type sequence is better than nothing or reversed sequence.
        }
    }

}
