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

        // Missing from original but used in code
        ATTACK, // Mapped to ATTACK_INITIATE
        BLOCK,  // Mapped to BLOCK_INITIATE
        PLAY,   // Mapped to PLAY_CARD
        CAST_SPELL, // New
        DESTROY, // New
        BREAK_SHIELD, // Mapped to SHIELD_BREAK

        // Custom/Other
        CUSTOM
    };

    struct GameEvent {
        EventType type;
        int source_id;         // instance_id of the source (or -1 if system)
        int target_id;         // instance_id of the target (or -1 if none)
        PlayerID player_id;    // Associated player (e.g., active player or owner)

        // Context for dynamic data
        std::map<std::string, int> context;

        GameEvent(EventType t, int src = -1, int tgt = -1, PlayerID pid = 255)
            : type(t), source_id(src), target_id(tgt), player_id(pid) {}
    };

}
