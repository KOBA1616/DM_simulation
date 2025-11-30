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

        // 6. PASS / RESOLVE_EFFECT / USE_SHIELD_TRIGGER (580 - 582)
        // We need to map these new actions to indices.
        // PASS is 580.
        // RESOLVE_EFFECT could be 581.
        // USE_SHIELD_TRIGGER could be 582.
        // But wait, USE_SHIELD_TRIGGER needs to know WHICH trigger if multiple?
        // ActionGenerator uses slot_index for these.
        // Let's expand the space slightly or map them.
        
        if (action.type == ActionType::PASS) {
            return offset;
        }
        offset += 1;

        if (action.type == ActionType::RESOLVE_EFFECT) {
             // Just one generic resolve action for now? 
             // Or do we need to distinguish?
             // ActionGenerator generates RESOLVE_EFFECT with slot_index.
             // But usually we resolve the top of stack or specific one.
             // If we have multiple pending, we might need selection.
             // For now, map to single index 581.
             return offset;
        }
        offset += 1;

        if (action.type == ActionType::USE_SHIELD_TRIGGER) {
            // Map to 582
            return offset;
        }
        offset += 1;
        
        return -1;
    }

}
