#ifndef DM_AI_ENCODERS_TOKEN_CONVERTER_HPP
#define DM_AI_ENCODERS_TOKEN_CONVERTER_HPP

#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../../engine/game_command/commands.hpp"
#include <vector>
#include <map>

namespace dm::ai::encoders {

    class TokenConverter {
    public:
        // Vocabulary Constants
        static constexpr int TOKEN_PAD = 0;
        static constexpr int TOKEN_CLS = 1;
        static constexpr int TOKEN_SEP = 2;
        static constexpr int TOKEN_UNK = 3;

        // Marker Base Offsets
        static constexpr int BASE_ZONE_MARKER = 10;
        static constexpr int BASE_STATE_MARKER = 50;
        static constexpr int BASE_PHASE_MARKER = 80; // New: To avoid collision with 0
        static constexpr int BASE_CONTEXT_MARKER = 100;
        static constexpr int BASE_COMMAND_MARKER = 200;
        static constexpr int BASE_CARD_ID = 1000;

        // Specific Markers (Relative to Perspective)
        static constexpr int MARKER_HAND_SELF = 10;
        static constexpr int MARKER_MANA_SELF = 11;
        static constexpr int MARKER_BATTLE_SELF = 12;
        static constexpr int MARKER_SHIELD_SELF = 13;
        static constexpr int MARKER_GRAVE_SELF = 14;
        static constexpr int MARKER_DECK_SELF = 15;

        static constexpr int MARKER_HAND_OPP = 20;
        static constexpr int MARKER_MANA_OPP = 21;
        static constexpr int MARKER_BATTLE_OPP = 22;
        static constexpr int MARKER_SHIELD_OPP = 23;
        static constexpr int MARKER_GRAVE_OPP = 24;
        static constexpr int MARKER_DECK_OPP = 25;

        static constexpr int STATE_TAPPED = 50;
        static constexpr int STATE_SICK = 51;
        static constexpr int STATE_FACE_DOWN = 52;

        static constexpr int CMD_TRANSITION = 200;
        static constexpr int CMD_MUTATE = 201;
        static constexpr int CMD_ATTACH = 202;
        static constexpr int CMD_FLOW = 203;
        static constexpr int CMD_QUERY = 204;
        static constexpr int CMD_DECIDE = 205;
        static constexpr int CMD_REACTION = 206;
        static constexpr int CMD_STAT = 207;
        static constexpr int CMD_RESULT = 208;

        /**
         * Encodes the game state and history into a token sequence.
         * @param state The game state.
         * @param perspective The player ID to view the game from (determines visible cards and Self/Opp markers).
         * @param max_len Max sequence length (0 = unlimited).
         * @return Vector of integer tokens.
         */
        static std::vector<int> encode_state(const dm::core::GameState& state, int perspective, int max_len = 0);

        static int get_vocab_size() { return 10000; }

    private:
        static void append_card(std::vector<int>& tokens, const dm::core::CardInstance& card, bool visible);
        static void append_zone(std::vector<int>& tokens, const std::vector<dm::core::CardInstance>& zone, int zone_token, bool visible);
        static void append_command_history(std::vector<int>& tokens, const dm::core::GameState& state, int limit);
    };

}

#endif // DM_AI_ENCODERS_TOKEN_CONVERTER_HPP
