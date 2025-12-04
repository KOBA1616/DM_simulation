#pragma once
#include "types.hpp"
#include "card_def.hpp"

namespace dm::core {

    enum class ActionType : uint8_t {
        PASS,
        MANA_CHARGE,
        PLAY_CARD,
        ATTACK_PLAYER,
        ATTACK_CREATURE,
        BLOCK,
        USE_SHIELD_TRIGGER,
        SELECT_TARGET,
        RESOLVE_EFFECT,
        USE_ABILITY, // For things like Revolution Change, Ninja Strike
        DECLARE_PLAY, // New
        PAY_COST,     // New
        RESOLVE_PLAY  // New
    };

    struct Action {
        ActionType type;
        CardID card_id; // For PLAY_CARD, MANA_CHARGE
        int source_instance_id; // For ATTACK, BLOCK (instance ID of the creature)
        int target_instance_id; // For ATTACK_CREATURE, SELECT_TARGET
        PlayerID target_player; // For ATTACK_PLAYER
        
        // ML Helper
        int slot_index = -1; // Index in Hand/BattleZone for source
        int target_slot_index = -1; // Index in BattleZone for target
        
        // For debugging
        std::string to_string() const {
            // Simplified string representation
            return "Action Type: " + std::to_string(static_cast<int>(type));
        }
    };

}
