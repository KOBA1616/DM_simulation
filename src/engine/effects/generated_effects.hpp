#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include <vector>
#include <algorithm>

namespace dm::engine {

    class GeneratedEffects {
    public:
        static void resolve(dm::core::GameState& game_state, const dm::core::PendingEffect& effect, int card_id) {
            auto& controller = game_state.players[effect.controller];
            auto& opponent = game_state.players[1 - effect.controller];

            // Previously-generated/per-card effect fallbacks have been migrated to JSON-driven GenericCardSystem.
            // Keep this function as an empty fallback to preserve compatibility for any legacy cases.
            (void)effect; (void)card_id; (void)controller; (void)opponent;
        }
    };
}
