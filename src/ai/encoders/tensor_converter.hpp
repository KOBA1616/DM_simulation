#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include <vector>
#include <map>

namespace dm::ai {

    class TensorConverter {
    public:
        // Returns the size of the input tensor
        // Updated to support Unified Masked/Full Representation
        // Structure:
        // [Global Features (10)]
        // [Self Player Features]
        //   - Hand Count (1)
        //   - Hand Cards (20)  <- Explicit slots
        //   - Mana (6)
        //   - Battle Zone (20 * 3)
        //   - Shield Count (1)
        //   - Graveyard (20)
        // [Opp Player Features]
        //   - Hand Count (1)
        //   - Hand Cards (20)  <- Explicit slots (Masked=0, Full=Values)
        //   - Mana (6)
        //   - Battle Zone (20 * 3)
        //   - Shield Count (1)
        //   - Graveyard (20)

        static constexpr int INPUT_SIZE = 
            10 + // Global (Turn, Phase, etc)
            (1 + 20 + 6 + 20 * 3 + 1 + 20) + // Self: HandCnt(1), Hand(20), ...
            (1 + 20 + 6 + 20 * 3 + 1 + 20);  // Opp: HandCnt(1), Hand(20), ...

        // Convert single state.
        // mask_opponent_hand: If true, opp hand slots are 0.0f. If false, filled with card info.
        static std::vector<float> convert_to_tensor(
            const dm::core::GameState& game_state,
            int player_view,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            bool mask_opponent_hand = true
        );
        
        // Batch conversion (Defaults to masked for standard training/inference usually,
        // but can be overridden. For now, we assume simple batch uses mask=true or exposes arg)
        static std::vector<float> convert_batch_flat(
            const std::vector<dm::core::GameState>& states,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            bool mask_opponent_hand = true
        );
    };

}
