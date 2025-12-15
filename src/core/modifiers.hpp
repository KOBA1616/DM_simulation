#pragma once
#include "types.hpp"
#include "card_json_types.hpp"
#include <string>
#include <optional>

namespace dm::core {

    struct CostModifier {
        int reduction_amount;
        // Condition/Filter for which cards this modifier applies to
        FilterDef condition_filter;
        int turns_remaining; // 1 = this turn only, >1 = persistent, -1 = indefinite
        int source_instance_id; // To track where it came from (e.g. Cocco Lupia)
        PlayerID controller;
    };

    // Step 5-1: Passive Effects
    enum class PassiveType {
        POWER_MODIFIER,
        // KEYWORD_GRANT was missing here, but present in handlers.
        // It must be defined for compilation.
        KEYWORD_GRANT,
        COST_REDUCTION, // Merged with CostModifier? Or keep separate?
        // CostModifier is for HAND/MANA cards. PassiveEffect is usually for BATTLE ZONE.
        // We will focus on BATTLE ZONE passives here.
        BLOCKER_GRANT,
        SPEED_ATTACKER_GRANT,
        SLAYER_GRANT,
        // Step 3-4: Attack/Block Restriction
        CANNOT_ATTACK,
        CANNOT_BLOCK,
        CANNOT_USE_SPELLS, // Step 3-x: "Cannot cast spell" (locking)
        LOCK_SPELL_BY_COST // "Declare Number -> Prohibit Spells"
    };

    struct PassiveEffect {
        PassiveType type;
        int value; // e.g. +1000 power
        std::string str_value; // e.g. "SPEED_ATTACKER"
        FilterDef target_filter; // Which creatures get this buff?
        ConditionDef condition; // "If shields 0..."
        int source_instance_id; // The source of the effect (e.g. Rose Castle)
        PlayerID controller;
        int turns_remaining = -1; // -1 = permanent, 1 = this turn only
    };

}
