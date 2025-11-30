#include "action_encoder.hpp"
#include <stdexcept>

namespace dm::ai {

    using namespace dm::core;

    int ActionEncoder::action_to_index(const Action& action) {
        int offset = 0;

        // 1. MANA_CHARGE
        if (action.type == ActionType::MANA_CHARGE) {
            if (action.slot_index >= 0 && action.slot_index < ACTION_MANA_SIZE) {
                return offset + action.slot_index;
            }
            return -1; // Invalid
        }
        offset += ACTION_MANA_SIZE;

        // 2. PLAY_CARD
        if (action.type == ActionType::PLAY_CARD) {
            if (action.slot_index >= 0 && action.slot_index < ACTION_PLAY_SIZE) {
                return offset + action.slot_index;
            }
            return -1;
        }
        offset += ACTION_PLAY_SIZE;

        // 3. ATTACK (split into ATTACK_PLAYER and ATTACK_CREATURE using MAX_BATTLE_SIZE)
        const int battle_slots = MAX_BATTLE_SIZE;
        const int attack_player_slots = battle_slots;
        const int attack_creature_slots = battle_slots * battle_slots;

        if (action.type == ActionType::ATTACK_PLAYER) {
            if (action.slot_index >= 0 && action.slot_index < attack_player_slots) {
                return offset + action.slot_index;
            }
            return -1;
        }
        offset += attack_player_slots;

        if (action.type == ActionType::ATTACK_CREATURE) {
            if (action.slot_index >= 0 && action.slot_index < battle_slots &&
                action.target_slot_index >= 0 && action.target_slot_index < battle_slots) {
                return offset + (action.slot_index * battle_slots) + action.target_slot_index;
            }
            return -1;
        }
        offset += attack_creature_slots;

        // 4. BLOCK
        if (action.type == ActionType::BLOCK) {
            if (action.slot_index >= 0 && action.slot_index < ACTION_BLOCK_SIZE) {
                return offset + action.slot_index;
            }
            return -1;
        }
        offset += ACTION_BLOCK_SIZE;

        // 5. SELECT_TARGET
        if (action.type == ActionType::SELECT_TARGET) {
            if (action.target_slot_index >= 0 && action.target_slot_index < ACTION_SELECT_TARGET_SIZE) {
                return offset + action.target_slot_index;
            }
            return -1;
        }
        offset += ACTION_SELECT_TARGET_SIZE;

        // 6. PASS, RESOLVE_EFFECT, USE_SHIELD_TRIGGER
        if (action.type == ActionType::PASS) {
            return offset; // single PASS index
        }
        offset += 1;

        if (action.type == ActionType::RESOLVE_EFFECT) {
            return offset; // single RESOLVE index
        }
        offset += 1;

        if (action.type == ActionType::USE_SHIELD_TRIGGER) {
            // If multiple triggers exist, Action.slot_index may disambiguate; currently map to this index.
            return offset;
        }
        offset += 1;

        return -1;
    }

}
