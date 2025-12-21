#pragma once
#include <string>
#include <vector>
#include <optional>
#include "card_json_types.hpp"
#include "types.hpp"

namespace dm::core {

    // Forward declarations
    struct CostModifier;
    struct PassiveEffect;

    // Phase 4: Cost System Structs
    // (Already defined in card_json_types.hpp for serialization,
    // but the runtime logic uses this structure in GameState.
    // Wait, GameState uses the one in game_state.hpp which matches card_json_types.hpp?
    // No, GameState uses `CostModifier` defined in `game_state.hpp`.
    // The `CostReductionDef` in `card_json_types.hpp` is for card definition.)

    struct CostModifier {
        int reduction_amount;
        // Condition/Filter for which cards this modifier applies to
        FilterDef condition_filter;
        int turns_remaining; // 1 = this turn only, >1 = persistent, -1 = indefinite
        int source_instance_id; // To track where it came from (e.g. Cocco Lupia)
        PlayerID controller;
        bool is_source_static = false; // Added for Continuous Effect System
    };

    enum class PassiveType {
        POWER_MODIFIER,
        KEYWORD_GRANT,
        COST_REDUCTION,
        BLOCKER_GRANT,
        SPEED_ATTACKER_GRANT,
        SLAYER_GRANT,
        CANNOT_ATTACK,
        CANNOT_BLOCK,
        CANNOT_USE_SPELLS,
        LOCK_SPELL_BY_COST,
        CANNOT_SUMMON
    };

    struct PassiveEffect {
        PassiveType type;
        int value; // e.g. +1000 power
        std::string str_value; // e.g. "SPEED_ATTACKER"
        FilterDef target_filter; // Which creatures get this buff?
        std::optional<std::vector<int>> specific_targets; // Specific instances (overrides filter if present)
        ConditionDef condition; // "If shields 0..."
        int source_instance_id; // The source of the effect
        PlayerID controller;
        int turns_remaining = -1; // -1 = permanent, 1 = this turn only
        bool is_source_static = false; // Added for Continuous Effect System
    };

}
