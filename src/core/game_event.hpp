#pragma once
#include "types.hpp"
#include <map>
#include <string>
#include <variant>

namespace dm::core {

    enum class EventType : uint8_t {
        NONE = 0,
        // Zone Changes
        ZONE_ENTER,       // Card enters a zone
        ZONE_LEAVE,       // Card leaves a zone

        // Phase/Turn
        TURN_START,
        TURN_END,
        PHASE_START,
        PHASE_END,

        // Actions
        PLAY_CARD,        // Card played (summon/cast)
        ATTACK_INITIATE,  // Attack declared
        BLOCK_INITIATE,   // Block declared
        BATTLE_START,     // Battle about to resolve
        BATTLE_WIN,
        BATTLE_LOSE,
        SHIELD_BREAK,
        DIRECT_ATTACK,

        // Tap/Untap
        TAP_CARD,
        UNTAP_CARD,

        // Custom/Other
        CUSTOM
    };

    struct GameEvent {
        EventType type;
        int instance_id;       // Replaces source_id for consistency with usage
        int card_id;           // Added card_id
        PlayerID player_id;    // Associated player (e.g., active player or owner)
        int target_id;         // instance_id of the target (or -1 if none)

        // Context for dynamic data
        std::map<std::string, int> context;

        GameEvent() : type(EventType::NONE), instance_id(-1), card_id(0), player_id(255), target_id(-1) {}

        GameEvent(EventType t, int inst = -1, int tgt = -1, PlayerID pid = 255)
            : type(t), instance_id(inst), card_id(0), player_id(pid), target_id(tgt) {}
    };

}
