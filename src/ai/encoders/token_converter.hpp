#pragma once

#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include <vector>
#include <map>
#include <optional>

namespace dm::ai::encoders {

    // Tokenized representation of the game state
    struct GameStateTokens {
        std::vector<int> global_features;
        std::vector<std::vector<int>> board_tokens;
        std::vector<std::vector<int>> history_tokens;
    };

    class TokenConverter {
    public:
        // Configuration
        static constexpr int MAX_HISTORY_LEN = 128;
        static constexpr int MAX_BOARD_ENTITIES = 64;
        static constexpr int VOCAB_SIZE = 4096;

        // Vocabulary Layout
        // 0-99: Special Tokens
        static constexpr int TOKEN_PAD = 0;
        static constexpr int TOKEN_CLS = 1; // Global Start
        static constexpr int TOKEN_SEP_BOARD = 2; // Board Start
        static constexpr int TOKEN_SEP_HISTORY = 3; // History Start

        // 100-299: Global Features / Status Buckets
        static constexpr int OFFSET_GLOBAL = 100;

        // 300-499: Status Flags & Properties
        // Tapped/Sick/Owner: 300-320
        // Power Buckets: 320-350
        // Cost Buckets: 360-380
        // Civ Bitmask: 390-455
        static constexpr int OFFSET_STATUS = 300;
        static constexpr int TOKEN_TAPPED = OFFSET_STATUS + 1;
        static constexpr int TOKEN_UNTAPPED = OFFSET_STATUS + 2;
        static constexpr int TOKEN_SICK = OFFSET_STATUS + 3;
        static constexpr int TOKEN_NOT_SICK = OFFSET_STATUS + 4;

        // 500-599: Action Types (Moved from 400 to avoid Civ overlap)
        static constexpr int OFFSET_ACTION = 500;

        // 1000+: Card IDs
        static constexpr int OFFSET_CARD_ID = 1000;
        static constexpr int MAX_CARD_ID_RANGE = 2500; // IDs 1000 to 3500

        static std::vector<int> encode_state(const core::GameState& state, int perspective = 0, int max_len = 0);
        static int get_vocab_size();

    private:
        static GameStateTokens tokenize_state(const core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
        static std::vector<int> tokenize_card_instance(const core::CardInstance& card, const core::CardDefinition& def);
        static std::vector<int> tokenize_command(const std::shared_ptr<dm::engine::game_command::GameCommand>& cmd);

        static int bucket_val(int val, int max_val);
    };

}
