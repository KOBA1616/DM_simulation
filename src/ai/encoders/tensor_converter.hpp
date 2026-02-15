#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include <vector>
#include <map>

namespace dm::ai {

    class TensorConverter {
    public:
        // Returns the size of the input tensor (Legacy ResNet)
        // Legacy input size expected by tests (kept for compatibility)
        static constexpr int INPUT_SIZE = 856;

        // Transformer V2 Constants
        static constexpr int MAX_SEQ_LEN = 200;
        static constexpr int VOCAB_SIZE = 1000;

        // Special Tokens
        enum SpecialToken {
            TOKEN_PAD = 0,
            TOKEN_SEP = 1,
            TOKEN_SELF_HAND_START = 2,
            TOKEN_SELF_MANA_START = 3,
            TOKEN_SELF_BATTLE_START = 4,
            TOKEN_SELF_GRAVE_START = 5,
            TOKEN_SELF_SHIELD_START = 6,
            TOKEN_OPP_HAND_START = 7,
            TOKEN_OPP_MANA_START = 8,
            TOKEN_OPP_BATTLE_START = 9,
            TOKEN_OPP_GRAVE_START = 10,
            TOKEN_OPP_SHIELD_START = 11,
            TOKEN_GLOBAL_START = 12,
            // Card IDs start from 100? or just map IDs?
            // Let's reserve 0-99 for special tokens.
            TOKEN_CARD_OFFSET = 100
        };

        // --- Legacy V1 (ResNet) ---
        static std::vector<float> convert_to_tensor(
            const dm::core::GameState& game_state,
            int player_view,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            bool mask_opponent_hand = true
        );

        // Overload for shared_ptr
        static std::vector<float> convert_batch_flat(
            const std::vector<std::shared_ptr<dm::core::GameState>>& states,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            bool mask_opponent_hand = true
        );
        
        static std::vector<float> convert_batch_flat(
            const std::vector<dm::core::GameState>& states,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            bool mask_opponent_hand = true
        );

        // --- Phase 4 V2 (Transformer) ---
        // Returns sequence of tokens [Batch, SeqLen]
        static std::vector<long> convert_to_sequence(
            const dm::core::GameState& game_state,
            int player_view,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            bool mask_opponent_hand = true
        );

        static std::vector<long> convert_batch_sequence(
            const std::vector<dm::core::GameState>& states,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            bool mask_opponent_hand = true
        );
    };

}
