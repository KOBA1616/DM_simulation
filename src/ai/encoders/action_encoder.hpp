#pragma once
#include "../../core/action.hpp"
#include "../../core/constants.hpp"

namespace dm::ai {

    class ActionEncoder {
    public:
        static constexpr int TOTAL_ACTION_SIZE = 
            dm::core::ACTION_MANA_SIZE + 
            dm::core::ACTION_PLAY_SIZE + 
            dm::core::ACTION_ATTACK_SIZE + 
            dm::core::ACTION_BLOCK_SIZE + 
            dm::core::ACTION_SELECT_TARGET_SIZE + 
            dm::core::ACTION_PASS_SIZE;

        // Maps an Action to a unique index in the policy vector
        static int action_to_index(const dm::core::Action& action);

        // Maps an index back to an Action (Partial, might need context to fully reconstruct)
        // Actually, for MCTS we usually just need Action -> Index to map policy logits to edges.
        // We don't necessarily need Index -> Action if we generate legal actions first and then map them.
    };

}
