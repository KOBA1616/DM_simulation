#pragma once
#include "core/card_json_types.hpp"
#include "core/constants.hpp"

namespace dm::ai {

    class ActionEncoder {
    public:
        static constexpr int TOTAL_ACTION_SIZE = 
            dm::core::ACTION_MANA_SIZE + 
            dm::core::ACTION_PLAY_SIZE + 
            dm::core::ACTION_ATTACK_SIZE + 
            dm::core::ACTION_BLOCK_SIZE + 
            dm::core::ACTION_SELECT_TARGET_SIZE + 
            dm::core::ACTION_PASS_SIZE + 
            10; // Extra buffer for RESOLVE, USE_SHIELD_TRIGGER etc.

        // Maps a CommandDef to a unique index in the policy vector
        static int action_to_index(const dm::core::CommandDef& cmd);
    };

}
