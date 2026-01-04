#include "engine/systems/breaker/breaker_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "core/modifiers.hpp"
#include <algorithm>

namespace dm::engine::systems {

    using namespace core;

    int BreakerSystem::get_breaker_count(const GameState& state, const CardInstance& creature, const CardDefinition& def) {
        int max_breaker = 1;

        // 1. Check Keywords on Card Definition
        if (def.keywords.world_breaker) max_breaker = 999;
        else if (def.keywords.triple_breaker) max_breaker = std::max(max_breaker, 3);
        else if (def.keywords.double_breaker) max_breaker = std::max(max_breaker, 2);

        // 2. Check Modifiers (KEYWORD_GRANT)
        for (const auto& eff : state.passive_effects) {
            if (eff.type == PassiveType::KEYWORD_GRANT) {
                // Check if effect applies to this creature
                if (TargetUtils::is_valid_target(creature, def, eff.target_filter, state, eff.controller, creature.owner, true)) {
                    if (eff.str_value == "WORLD_BREAKER") max_breaker = 999;
                    else if (eff.str_value == "TRIPLE_BREAKER") max_breaker = std::max(max_breaker, 3);
                    else if (eff.str_value == "DOUBLE_BREAKER") max_breaker = std::max(max_breaker, 2);
                }
            }
        }

        return max_breaker;
    }

}
