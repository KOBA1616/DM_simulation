#include "action_encoder.hpp"
#include <stdexcept>

namespace dm::ai {

    using namespace dm::core;

    int ActionEncoder::action_to_index(const CommandDef& cmd) {
        int offset = 0;

        // 1. MANA_CHARGE
        if (cmd.type == CommandType::MANA_CHARGE) {
            if (cmd.slot_index >= 0 && cmd.slot_index < ACTION_MANA_SIZE) {
                return offset + cmd.slot_index;
            }
            return -1; // Invalid
        }
        offset += ACTION_MANA_SIZE;

        // 2. PLAY_FROM_ZONE
        if (cmd.type == CommandType::PLAY_FROM_ZONE) {
            if (cmd.slot_index >= 0 && cmd.slot_index < ACTION_PLAY_SIZE) {
                return offset + cmd.slot_index;
            }
            return -1;
        }
        offset += ACTION_PLAY_SIZE;

        // 3. ATTACK (split into ATTACK_PLAYER and ATTACK_CREATURE using MAX_BATTLE_SIZE)
        const int battle_slots = MAX_BATTLE_SIZE;
        const int attack_player_slots = battle_slots;
        const int attack_creature_slots = battle_slots * battle_slots;

        if (cmd.type == CommandType::ATTACK_PLAYER) {
            if (cmd.slot_index >= 0 && cmd.slot_index < attack_player_slots) {
                return offset + cmd.slot_index;
            }
            return -1;
        }
        offset += attack_player_slots;

        if (cmd.type == CommandType::ATTACK_CREATURE) {
            if (cmd.slot_index >= 0 && cmd.slot_index < battle_slots &&
                cmd.target_slot_index >= 0 && cmd.target_slot_index < battle_slots) {
                return offset + (cmd.slot_index * battle_slots) + cmd.target_slot_index;
            }
            return -1;
        }
        offset += attack_creature_slots;

        // 4. BLOCK
        if (cmd.type == CommandType::BLOCK) {
            if (cmd.slot_index >= 0 && cmd.slot_index < ACTION_BLOCK_SIZE) {
                return offset + cmd.slot_index;
            }
            return -1;
        }
        offset += ACTION_BLOCK_SIZE;

        // 5. SELECT_TARGET
        if (cmd.type == CommandType::SELECT_TARGET) {
            if (cmd.target_slot_index >= 0 && cmd.target_slot_index < ACTION_SELECT_TARGET_SIZE) {
                return offset + cmd.target_slot_index;
            }
            return -1;
        }
        offset += ACTION_SELECT_TARGET_SIZE;

        // 6. PASS, RESOLVE_EFFECT, SHIELD_TRIGGER
        if (cmd.type == CommandType::PASS) {
            return offset; // single PASS index
        }
        offset += 1;

        if (cmd.type == CommandType::RESOLVE_EFFECT) {
            return offset; // single RESOLVE index
        }
        offset += 1;

        if (cmd.type == CommandType::SHIELD_TRIGGER) {
            return offset;
        }
        offset += 1;

        return -1;
    }

}
