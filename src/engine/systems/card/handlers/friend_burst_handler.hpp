#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include "../../../../core/card_def.hpp"
#include "../card_registry.hpp"
#include "../target_utils.hpp"
#include <iostream>

namespace dm::engine {

    class FriendBurstHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            (void)ctx;
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            if (!ctx.targets || ctx.targets->empty()) return;

            int target_id = (*ctx.targets)[0];
            CardInstance* target_creature = ctx.game_state.get_card_instance(target_id);
            if (!target_creature) return;

            // 2. Tap target
            target_creature->is_tapped = true;

            // 3. Resolve Spell Side
            CardInstance* source_card = ctx.game_state.get_card_instance(ctx.source_instance_id);
            if (!source_card) return;

            if (ctx.card_db.count(source_card->card_id)) {
                const auto& data = ctx.card_db.at(source_card->card_id);
                if (data.spell_side) {
                    const auto& spell_def = *data.spell_side;

                    ctx.game_state.turn_stats.spells_cast_this_turn++;

                    for (const auto& effect : spell_def.effects) {
                        GenericCardSystem::resolve_effect_with_context(ctx.game_state, effect, ctx.source_instance_id, ctx.execution_vars, ctx.card_db);
                    }
                }
            }
        }
    };
}
