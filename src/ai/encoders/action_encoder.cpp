#include "action_encoder.hpp"
#include <stdexcept>

namespace dm::ai {

    using namespace dm::core;

    int ActionEncoder::action_to_index(const Action& action) {
        int offset = 0;

        // 1. MANA_CHARGE (0 - 19)
        if (action.type == ActionType::MANA_CHARGE) {
            if (action.slot_index >= 0 && action.slot_index < ACTION_MANA_SIZE) {
                return offset + action.slot_index;
            }
            return -1; // Invalid
        }
        offset += ACTION_MANA_SIZE;

        // 2. PLAY_CARD (20 - 39)
        if (action.type == ActionType::PLAY_CARD) {
            if (action.slot_index >= 0 && action.slot_index < ACTION_PLAY_SIZE) {
                return offset + action.slot_index;
            }
            return -1;
        }
        offset += ACTION_PLAY_SIZE;

        // 3. ATTACK (40 - 459)
        // Attack Player: 20 slots (one for each creature slot)
        // Attack Creature: 20 * 20 = 400 slots (source slot * target slot)
        // Total 420
        if (action.type == ActionType::ATTACK_PLAYER) {
            if (action.slot_index >= 0 && action.slot_index < 20) {
                return offset + action.slot_index;
            }
            return -1;
        }
        offset += 20; // Attack Player slots

        if (action.type == ActionType::ATTACK_CREATURE) {
            if (action.slot_index >= 0 && action.slot_index < 20 && 
                action.target_slot_index >= 0 && action.target_slot_index < 20) {
                return offset + (action.slot_index * 20) + action.target_slot_index;
            }
            return -1;
        }
        offset += 400; // Attack Creature slots

        // 4. BLOCK (460 - 479)
        if (action.type == ActionType::BLOCK) {
            if (action.slot_index >= 0 && action.slot_index < ACTION_BLOCK_SIZE) {
                return offset + action.slot_index;
            }
            return -1;
        }
        offset += ACTION_BLOCK_SIZE;

        // 5. SELECT_TARGET (480 - 579)
        if (action.type == ActionType::SELECT_TARGET) {
            // Assuming target_slot_index maps to 0-99
            if (action.target_slot_index >= 0 && action.target_slot_index < ACTION_SELECT_TARGET_SIZE) {
                return offset + action.target_slot_index;
            }
            return -1;
        }
        offset += ACTION_SELECT_TARGET_SIZE;

        // 6. PASS (580)
        if (action.type == ActionType::PASS) {
            return offset;
        }
        
        return -1;
    }

}
