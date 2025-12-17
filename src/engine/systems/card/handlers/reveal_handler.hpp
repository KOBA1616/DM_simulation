#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include <algorithm>
#include <iostream>

namespace dm::engine {

    class RevealHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            // Stub for simple resolve
            (void)ctx;
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             if (!ctx.targets) return;
             for (int id : *ctx.targets) {
                 dm::core::CardInstance* card = ctx.game_state.get_card_instance(id);
                 if (card) {
                     ctx.game_state.on_card_reveal(card->card_id);
                 }
             }
        }
    };
}
