#pragma once
#include "types.hpp"
#include "card_def.hpp"

namespace dm::core {

    enum class ActionType : uint8_t {
        PASS,
        MANA_CHARGE,
        MOVE_CARD,
        PLAY_CARD,
        ATTACK_PLAYER,
        ATTACK_CREATURE,
        BLOCK,
        USE_SHIELD_TRIGGER,
        SELECT_TARGET,
        RESOLVE_EFFECT,
        USE_ABILITY, // For things like Revolution Change, Ninja Strike
        DECLARE_PLAY, // Atomic
        PAY_COST,     // Atomic
        RESOLVE_PLAY, // Atomic
        PLAY_CARD_INTERNAL, // For stacked play actions
        RESOLVE_BATTLE, // Battle resolution (Power comparison)
        BREAK_SHIELD,   // Shield break
        DECLARE_REACTION, // For Ninja Strike / Strike Back
        SELECT_OPTION, // For choosing from options
        SELECT_NUMBER // For choosing a number
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

        // Spawn Source for PLAY_CARD_INTERNAL
        SpawnSource spawn_source = SpawnSource::HAND_SUMMON;
        
        // Step 3-3: Destination Override
        // 0 = Default (Grave for spell, Battle for creature), 1 = Deck Bottom, 2 = Deck Top, 3 = Hand, 4 = Mana, 5 = Shield
        uint8_t destination_override = 0;

        // Step 4-1: Twinpact Support
        bool is_spell_side = false;

        // For debugging
        std::string to_string() const {
            // Simplified string representation
            return "Action Type: " + std::to_string(static_cast<int>(type));
        }
    };

}
